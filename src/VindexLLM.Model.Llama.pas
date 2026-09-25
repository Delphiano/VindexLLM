{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Native LLaMA-family decoder support.  This implementation covers the GGUF
  layout used by TinyLlama 1.1B Chat v1.0 (architecture = "llama").
===============================================================================}

unit VindexLLM.Model.Llama;

{$I VindexLLM.Defines.inc}

interface

uses
  VindexLLM.GGUFReader,
  VindexLLM.WeightPager,
  VindexLLM.Model,
  VindexLLM.Vulkan,
  VindexLLM.Compute;

type
  TVdxLlamaModel = class(TVdxModel)
  private
    FPager: TVdxWeightPager;
    FOutputGpu: TVdxGpuBuffer;
    FOutputType: TVdxGGMLType;
    FSwigluShader: VkShaderModule;
    FSwigluBundle: TVdxComputePipelineBundle;
    FSwigluDescLayout: VkDescriptorSetLayout;
    FSwigluDescPool: VkDescriptorPool;
    FSwigluDescSet: VkDescriptorSet;
    procedure CreateIdentityNorm(const ACount: UInt32;
      out ABuffer: TVdxGpuBuffer);
    procedure InitSwigluKernel();
    procedure FreeSwigluKernel();
    procedure ApplySiluMul(const AGate, AUp: TVdxGpuBuffer;
      const ACount: UInt32);
    procedure AddToResidual(const AInput: TVdxGpuBuffer;
      const ACount: UInt32; const ABatch: Boolean);
  public
    constructor Create(); override;
    destructor Destroy(); override;

    class function SupportedArchitectures(): TArray<string>; override;
    function LoadModelConfig(const AReader: TVdxGGUFReader;
      const AMaxContext: Integer): Boolean; override;
    function InitSubsystems(): Boolean; override;
    function LoadWeights(): Boolean; override;
    procedure FreeWeights(); override;
    function PrefillBatchSize(): Integer; override;
    function AllocatedBytes(var AWeights, ACache, AScratch: UInt64): Boolean; override;

    procedure RunLayerForward(const ALayer: Integer;
      const APosition: Integer); override;
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens: UInt32; const AStartPos: UInt32;
      const ABidirectional: Boolean = False); override;
    procedure UnembedToLogits(const AOutLogits: TVdxGpuBuffer); override;

    function FormatPrompt(const APrompt: string): string; override;
    function GetStopTokenStrings(): TArray<string>; override;
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  System.Classes,
  Winapi.Windows,
  VindexLLM.Utils,
  VindexLLM.Attention,
  VindexLLM.LayerNorm,
  VindexLLM.Model.Ministral3,
  VindexLLM.Model.Registry;

constructor TVdxLlamaModel.Create();
begin
  inherited;
  FPager := nil;
  FOutputGpu := Default(TVdxGpuBuffer);
  FOutputType := gtF16;
  FSwigluShader := VK_NULL_HANDLE;
  FSwigluBundle := Default(TVdxComputePipelineBundle);
  FSwigluDescLayout := VK_NULL_HANDLE;
  FSwigluDescPool := VK_NULL_HANDLE;
  FSwigluDescSet := VK_NULL_HANDLE;
end;

destructor TVdxLlamaModel.Destroy();
begin
  FreeSwigluKernel();
  inherited;
end;

class function TVdxLlamaModel.SupportedArchitectures(): TArray<string>;
begin
  Result := ['llama'];
end;

function TVdxLlamaModel.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
var
  LQInfo, LEmbedInfo: TVdxGGUFTensorInfo;
  LModelMax: UInt32;
begin
  Result := False;
  if not inherited LoadModelConfig(AReader, AMaxContext) then Exit;

  FNumLayers := AReader.GetMetadataUInt32('llama.block_count');
  FHiddenDim := AReader.GetMetadataUInt32('llama.embedding_length');
  FFFNWidth := AReader.GetMetadataUInt32('llama.feed_forward_length');
  FNumQHeads := AReader.GetMetadataUInt32('llama.attention.head_count');
  FNumKVHeads := AReader.GetMetadataUInt32('llama.attention.head_count_kv');
  if (FNumLayers = 0) or (FHiddenDim = 0) or (FFFNWidth = 0) or
     (FNumQHeads = 0) or (FNumKVHeads = 0) then
  begin
    FErrors.Add(esFatal, 'CONF', 'Invalid or incomplete LLaMA GGUF metadata');
    Exit;
  end;

  if not AReader.GetTensorInfo('blk.0.attn_q.weight', LQInfo) then
  begin
    FErrors.Add(esFatal, 'CONF', 'Missing required tensor: blk.0.attn_q.weight');
    Exit;
  end;
  FHeadDim := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;
  FWeightType := LQInfo.TensorType;
  if (FHeadDim = 0) or (FHeadDim mod 32 <> 0) then
  begin
    FErrors.Add(esFatal, 'CONF', 'Unsupported LLaMA attention head size: %d', [FHeadDim]);
    Exit;
  end;

  if not AReader.GetTensorInfo('token_embd.weight', LEmbedInfo) then
  begin
    FErrors.Add(esFatal, 'CONF', 'Missing required tensor: token_embd.weight');
    Exit;
  end;
  FEmbedType := LEmbedInfo.TensorType;

  LModelMax := AReader.GetMetadataUInt32('llama.context_length');
  if LModelMax = 0 then LModelMax := 2048;
  if AMaxContext <= 0 then FMaxSeqLen := LModelMax
  else FMaxSeqLen := Min(UInt32(AMaxContext), LModelMax);

  Status('Native LLaMA: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d context=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads,
     FHeadDim, FMaxSeqLen]);
  Result := True;
end;

function TVdxLlamaModel.InitSubsystems(): Boolean;
begin
  Result := inherited InitSubsystems();
  if Result then InitSwigluKernel();
  Result := Result and not FErrors.HasFatal();
end;

procedure TVdxLlamaModel.InitSwigluKernel();
var
  LStream: TResourceStream;
  LSpv: TBytes;
begin
  // Ministral's embedded dense kernel contains operation 8: SiLU(a) * b.
  // Keeping a single copy avoids another toolchain or runtime dependency.
  LStream := TResourceStream.Create(HInstance, 'MINISTRAL_DENSE', RT_RCDATA);
  try
    SetLength(LSpv, LStream.Size);
    LStream.ReadBuffer(LSpv[0], LStream.Size);
  finally
    LStream.Free();
  end;
  FSwigluShader := FCompute.CreateShaderModule(@LSpv[0], NativeUInt(Length(LSpv)));
  FSwigluDescLayout := FCompute.CreateStorageDescriptorSetLayout(4);
  FSwigluBundle := FCompute.CreateComputePipelineWithPush(FSwigluShader, 'main',
    FSwigluDescLayout, SizeOf(TVdxMinistralPush));
end;

procedure TVdxLlamaModel.FreeSwigluKernel();
begin
  if FCompute = nil then Exit;
  if FSwigluDescPool <> VK_NULL_HANDLE then FCompute.DestroyDescriptorPoolHandle(FSwigluDescPool);
  if FSwigluBundle.Pipeline <> VK_NULL_HANDLE then FCompute.DestroyComputePipelineBundle(FSwigluBundle);
  if FSwigluDescLayout <> VK_NULL_HANDLE then FCompute.DestroyDescriptorSetLayoutHandle(FSwigluDescLayout);
  if FSwigluShader <> VK_NULL_HANDLE then FCompute.DestroyShaderModuleHandle(FSwigluShader);
  FSwigluDescPool := VK_NULL_HANDLE;
  FSwigluDescSet := VK_NULL_HANDLE;
  FSwigluDescLayout := VK_NULL_HANDLE;
  FSwigluBundle := Default(TVdxComputePipelineBundle);
  FSwigluShader := VK_NULL_HANDLE;
end;

procedure TVdxLlamaModel.CreateIdentityNorm(const ACount: UInt32;
  out ABuffer: TVdxGpuBuffer);
var
  LValues: array of Single;
  I: Integer;
begin
  ABuffer := Default(TVdxGpuBuffer);
  SetLength(LValues, ACount);
  for I := 0 to Integer(ACount) - 1 do LValues[I] := 1.0;
  ABuffer := FCompute.CreateGpuBuffer(UInt64(ACount) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FCompute.UploadToBuffer(ABuffer, @LValues[0], UInt64(ACount) * SizeOf(Single));
end;

function TVdxLlamaModel.LoadWeights(): Boolean;
var
  LLayer, LSlot: Integer;
  LPrefix: string;
  LBudget: UInt64;
  LOutputInfo: TVdxGGUFTensorInfo;
const
  Names: array[0..6] of string = ('attn_q.weight', 'attn_k.weight',
    'attn_v.weight', 'attn_output.weight', 'ffn_gate.weight',
    'ffn_up.weight', 'ffn_down.weight');
begin
  Result := False;
  if not FFFN.BuildFromGGUF(FReader) then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Failed to build LLaMA FFN index from GGUF');
    Exit;
  end;
  FPager := TVdxWeightPager.Create(FCompute, FReader);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    for LSlot := 0 to High(Names) do
      FPager.Add(Format('blk.%d.%s', [LLayer, Names[LSlot]]), LLayer, LSlot);

  SetLength(FAttnWeights, FNumLayers);
  SetLength(FUpWeights, FNumLayers);
  SetLength(FNormWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    FNormWeights[LLayer].AttnNormGpu := UploadNormWeight(
      Format('blk.%d.attn_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].FFNNormGpu := UploadNormWeight(
      Format('blk.%d.ffn_norm.weight', [LLayer]), FHiddenDim);
    // LLaMA has no Q/K RMSNorm. Nil handles disable that optional pass.
  end;
  FOutputNormGpu := UploadNormWeight('output_norm.weight', FHiddenDim);
  if FErrors.HasFatal() then Exit;

  FEmbedGpu := UploadWeightTensor('token_embd.weight');
  if FReader.HasTensor('output.weight') then
  begin
    FOutputGpu := UploadWeightTensor('output.weight');
    if not FReader.GetTensorInfo('output.weight', LOutputInfo) then
    begin
      FErrors.Add(esFatal, 'LOAD', 'Cannot inspect output.weight');
      Exit;
    end;
    FOutputType := LOutputInfo.TensorType;
  end
  else
  begin
    FOutputGpu := FEmbedGpu;
    FOutputType := FEmbedType;
  end;
  FEmbedScale := 1.0;
  if FErrors.HasFatal() then Exit;
  if not BuildBatchResources() then Exit;

  // Bind the FFN scratch buffers only after all model resources are allocated.
  FSwigluDescPool := FCompute.CreateDescriptorPoolForStorage(1, 4);
  FSwigluDescSet := FCompute.AllocateDescriptorSetForBuffers(FSwigluDescPool,
    FSwigluDescLayout, [FGateBuf, FUpBuf, FUpBuf, FGateBuf]);
  if FErrors.HasFatal() then Exit;

  LBudget := (FCompute.GetVRAMSizeMB() * UInt64(1024 * 1024) div 4) * 3;
  FPager.Allocate(FNumLayers, LBudget);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    LPrefix := Format('blk.%d.', [LLayer]);
    FAttnWeights[LLayer].WeightType := FWeightType;
    FAttnWeights[LLayer].QWeightGpu := FPager.Buffer(LPrefix + Names[0]);
    FAttnWeights[LLayer].KWeightGpu := FPager.Buffer(LPrefix + Names[1]);
    FAttnWeights[LLayer].VWeightGpu := FPager.Buffer(LPrefix + Names[2]);
    FAttnWeights[LLayer].OWeightGpu := FPager.Buffer(LPrefix + Names[3]);
    FFFN.SetLayerBuffers(LLayer, FPager.Buffer(LPrefix + Names[4]),
      FPager.Buffer(LPrefix + Names[6]));
    FUpWeights[LLayer] := FPager.Buffer(LPrefix + Names[5]);
  end;
  Status('LLaMA weights loaded: %d resident layer(s), %d streamed',
    [FPager.ResidentLayers, Integer(FNumLayers) - FPager.ResidentLayers]);
  Result := True;
end;

procedure TVdxLlamaModel.FreeWeights();
var
  I: Integer;
  Empty: TVdxGpuBuffer;
begin
  if FPager <> nil then
  begin
    FAttnWeights := nil;
    FUpWeights := nil;
    Empty := Default(TVdxGpuBuffer);
    for I := 0 to FFFN.GetLayerCount() - 1 do FFFN.SetLayerBuffers(I, Empty, Empty);
    FreeAndNil(FPager);
  end;
  // output may alias the embedding table when a GGUF has tied embeddings.
  if (FOutputGpu.Buffer <> VK_NULL_HANDLE) and (FOutputGpu.Buffer <> FEmbedGpu.Buffer) then
    FCompute.DestroyGpuBuffer(FOutputGpu);
  FOutputGpu := Default(TVdxGpuBuffer);
  inherited;
end;

function TVdxLlamaModel.PrefillBatchSize(): Integer;
begin
  Result := Min(Integer(FMaxSeqLen), 32);
end;

function TVdxLlamaModel.AllocatedBytes(var AWeights, ACache, AScratch: UInt64): Boolean;
var
  I: Integer;
begin
  Result := FPager <> nil;
  if not Result then Exit;
  AWeights := FPager.WeightBytes + FEmbedGpu.AllocationBytes + FOutputNormGpu.AllocationBytes;
  if FOutputGpu.Buffer <> FEmbedGpu.Buffer then Inc(AWeights, FOutputGpu.AllocationBytes);
  for I := 0 to High(FNormWeights) do
    Inc(AWeights, FNormWeights[I].AttnNormGpu.AllocationBytes +
      FNormWeights[I].FFNNormGpu.AllocationBytes +
      FNormWeights[I].QNormGpu.AllocationBytes + FNormWeights[I].KNormGpu.AllocationBytes);
  ACache := UInt64(2) * FNumLayers * CacheBytesPerLayer();
  AScratch := FCompute.AllocatedBytes - AWeights - ACache;
end;

procedure TVdxLlamaModel.ApplySiluMul(const AGate, AUp: TVdxGpuBuffer;
  const ACount: UInt32);
var
  LPush: TVdxMinistralPush;
begin
  LPush := Default(TVdxMinistralPush);
  LPush.Op := 8; // SwiGLU: SiLU(A) * B
  LPush.InDim := ACount;
  LPush.Tokens := 1;
  FCompute.UpdateDescriptorSetBuffers(FSwigluDescSet, [AGate, AUp, AUp, AGate]);
  FCompute.DispatchComputeWithPush(FSwigluBundle.Pipeline, FSwigluBundle.PipelineLayout,
    FSwigluDescSet, @LPush, SizeOf(LPush), (ACount + 255) div 256);
end;

procedure TVdxLlamaModel.AddToResidual(const AInput: TVdxGpuBuffer;
  const ACount: UInt32; const ABatch: Boolean);
var
  LPush: TVdxVecAddPush;
begin
  LPush.Count := ACount;
  if ABatch then
  begin
    FCompute.UpdateDescriptorSetBuffers(FBatchVecAddAttnDescSet, [FResidualMat, AInput]);
    FCompute.DispatchComputeWithPush(FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
      FBatchVecAddAttnDescSet, @LPush, SizeOf(LPush), (ACount + 255) div 256);
  end
  else
  begin
    FCompute.UpdateDescriptorSetBuffers(FVecAddAttnDescSet, [FResidualGpu, AInput]);
    FCompute.DispatchComputeWithPush(FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
      FVecAddAttnDescSet, @LPush, SizeOf(LPush), (ACount + 255) div 256);
  end;
end;

procedure TVdxLlamaModel.RunLayerForward(const ALayer: Integer; const APosition: Integer);
begin
  if FPager <> nil then FPager.PrepareLayer(ALayer);
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].AttnNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();
  FAttn.Forward(FWorkBufA, FAttnWeights[ALayer], FNormWeights[ALayer].QNormGpu,
    FNormWeights[ALayer].KNormGpu, ALayer, APosition, 10000.0, FAttnOutBuf);
  AddToResidual(FAttnOutBuf, FHiddenDim, False);
  FCompute.BatchBarrier();
  FNorm.ApplyCopy(FResidualGpu, FNormWeights[ALayer].FFNNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();
  FAttn.TestMatVec(FFFN.GetLayer(ALayer).GateGpuBuffer, FWorkBufA, FGateBuf,
    FHiddenDim, FFFNWidth, FWeightType);
  FAttn.TestMatVec(FUpWeights[ALayer], FWorkBufA, FUpBuf, FHiddenDim, FFFNWidth, FWeightType);
  FCompute.BatchBarrier();
  ApplySiluMul(FGateBuf, FUpBuf, FFFNWidth);
  FCompute.BatchBarrier();
  FAttn.TestMatVec(FFFN.GetLayer(ALayer).DownGpuBuffer, FGateBuf, FFFNOutBuf,
    FFFNWidth, FHiddenDim, FWeightType);
  FCompute.BatchBarrier();
  AddToResidual(FFFNOutBuf, FHiddenDim, False);
  FCompute.BatchBarrier();
end;

procedure TVdxLlamaModel.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens: UInt32; const AStartPos: UInt32; const ABidirectional: Boolean);
begin
  if ABidirectional then raise ENotSupportedException.Create('LLaMA is a causal decoder model');
  if FPager <> nil then FPager.PrepareLayer(ALayer);
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].AttnNormGpu, FWorkMat,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();
  FAttn.ForwardBatch(FWorkMat, FAttnWeights[ALayer], FNormWeights[ALayer].QNormGpu,
    FNormWeights[ALayer].KNormGpu, ALayer, ANumTokens, AStartPos, 10000.0,
    FQMat, FKMat, FVMat, FAttnOutMatBuf, False);
  AddToResidual(FAttnOutMatBuf, ANumTokens * FHiddenDim, True);
  FCompute.BatchBarrier();
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].FFNNormGpu, FWorkMat,
    FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();
  FAttn.BatchMatMul(FFFN.GetLayer(ALayer).GateGpuBuffer, FWorkMat, FGateMat,
    FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FAttn.BatchMatMul(FUpWeights[ALayer], FWorkMat, FUpMatBuf,
    FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FCompute.BatchBarrier();
  ApplySiluMul(FGateMat, FUpMatBuf, ANumTokens * FFFNWidth);
  FCompute.BatchBarrier();
  FAttn.BatchMatMul(FFFN.GetLayer(ALayer).DownGpuBuffer, FGateMat, FFFNOutMat,
    FFFNWidth, FHiddenDim, ANumTokens, FWeightType);
  FCompute.BatchBarrier();
  AddToResidual(FFFNOutMat, ANumTokens * FHiddenDim, True);
  FCompute.BatchBarrier();
end;

procedure TVdxLlamaModel.UnembedToLogits(const AOutLogits: TVdxGpuBuffer);
begin
  FCompute.BeginBatch();
  FNorm.ApplyCopy(FResidualGpu, FOutputNormGpu, FWorkBufA, FHiddenDim);
  FCompute.BatchBarrier();
  FAttn.TestMatVec(FOutputGpu, FWorkBufA, AOutLogits, FHiddenDim,
    UInt32(FVocabSize), FOutputType);
  FCompute.EndBatch();
end;

function TVdxLlamaModel.FormatPrompt(const APrompt: string): string;
begin
  // TinyLlama Chat v1.0 was trained with an end-of-turn token after each
  // user message.  Omitting it makes the assistant header part of the user
  // turn, which can produce degenerate output (commonly repeated <unk>).
  Result := '<|user|>' + #10 + APrompt + '</s>' + #10 +
    '<|assistant|>' + #10;
end;

function TVdxLlamaModel.GetStopTokenStrings(): TArray<string>;
begin
  Result := ['</s>'];
end;

initialization
  TVdxModelRegistry.RegisterClass(TVdxLlamaModel);

end.
