unit VindexLLM.Model.Qwen35;

interface

uses
  System.SysUtils, System.Generics.Collections,
  VindexLLM.Model, VindexLLM.GGUFReader, VindexLLM.Compute, VindexLLM.Vulkan;

type
  PVdxQwen35Buffer = ^TVdxGpuBuffer;
  // Fields match the GLSL push-constant block (72 bytes, all scalar 32-bit).
  TVdxQwen35Push = record
    Op, InDim, OutDim, TensorType: UInt32;
    Tokens, StartPos, MaxSeq, Heads, KVHeads, HeadDim: UInt32;
    Eps, Theta, FreqScale, CorrLow, CorrHigh, Magnitude, TempScale: Single;
    OrigContext: UInt32;
  end;

  TVdxQwen35Tensor = record
    Buffer: TVdxGpuBuffer;
    Kind: TVdxGGMLType;
    Width, Rows: UInt32;
  end;

  TVdxQwen35Layer = record
    Q, K, V, O, Gate, Up, Down, AttnNorm, FFNNorm: TVdxQwen35Tensor;
    QNorm, KNorm, Conv, Alpha, Beta, Decay, DT, SNorm, Z: TVdxQwen35Tensor;
    KCache, VCache: TVdxGpuBuffer;
  end;

  TVdxQwen35Model = class(TVdxModel)
  private
    FLayers: TArray<TVdxQwen35Layer>;
    FEmbedding, FOutput, FFinalNorm: TVdxQwen35Tensor;
    FOwned: TList<TVdxGpuBuffer>;
    FLayout: VkDescriptorSetLayout;
    FShader: VkShaderModule;
    FKernel: TVdxComputePipelineBundle;
    FParams: TVdxQwen35Push;
    FQ, FK, FV, FWork, FAttended, FProjected, FGate, FUp, FScores: TVdxGpuBuffer;
    FTokenIds, FDummy, FQGate, FConv, FZ, FAlpha, FBeta, FDecay: TVdxGpuBuffer;
    FWeightBytes, FCacheBytes, FScratchBytes: UInt64;
    FBatchCount: UInt32;
    function NewBuffer(const ABytes: UInt64; const AHost: Boolean = False): TVdxGpuBuffer;
    function ReadTensor(const AName: string; const AWidth, ARows: UInt32;
      const ANorm: Boolean = False): TVdxQwen35Tensor;
    procedure DispatchKernel(const APush: TVdxQwen35Push;
      const A, B, C, D: TVdxGpuBuffer; const AX: UInt32;
      const AY: UInt32 = 1; const AZ: UInt32 = 1; const AState: PVdxQwen35Buffer = nil);
    procedure MatMul(const W: TVdxQwen35Tensor; const X, Y: TVdxGpuBuffer; const N: UInt32);
    procedure Norm(const X: TVdxGpuBuffer; const W: TVdxQwen35Tensor;
      const Y: TVdxGpuBuffer; const N: UInt32);
    procedure Layer(const ALayer: Integer; const N, Position: UInt32);
  public
    constructor Create(); override;
    destructor Destroy(); override;
    class function SupportedArchitectures(): TArray<string>; override;
    function LoadModelConfig(const AReader: TVdxGGUFReader; const AMaxContext: Integer): Boolean; override;
    function InitSubsystems(): Boolean; override;
    function LoadWeights(): Boolean; override;
    procedure FreeWeights(); override;
    procedure RunLayerForward(const ALayer, APosition: Integer); override;
    procedure RunLayerForwardBatch(const ALayer: Integer; const ANumTokens, AStartPos: UInt32;
      const ABidirectional: Boolean = False); override;
    procedure EmbedToken(const ATokenId: Integer); override;
    procedure EmbedTokensBatch(const ATokenIds: TArray<Integer>; const ANumTokens: Integer;
      const AOutputBuf: TVdxGpuBuffer); override;
    procedure SeedResidualFromBatchLast(const ANumTokens: UInt32); override;
    procedure UnembedToLogits(const AOutLogits: TVdxGpuBuffer); override;
    function FormatPrompt(const APrompt: string): string; override;
    function GetStopTokenStrings(): TArray<string>; override;
    function PrefillBatchSize(): Integer; override;
    function CacheFormat(): UInt32; override;
    function CacheBytesPerLayer(): UInt64; override;
    function CacheBuffer(const ALayer: Integer; const AKey: Boolean): TVdxGpuBuffer; override;
    function AllocatedBytes(var AWeights, ACache, AScratch: UInt64): Boolean; override;
  end;

implementation

uses
  System.Math, System.Classes, Winapi.Windows,
  VindexLLM.Utils, VindexLLM.VirtualBuffer, VindexLLM.Model.Registry;

{$R qwen35\Qwen35.res}

constructor TVdxQwen35Model.Create();
begin
  inherited;
  FOwned := TList<TVdxGpuBuffer>.Create();
end;

destructor TVdxQwen35Model.Destroy();
begin
  FreeWeights();
  FreeAndNil(FOwned);
  inherited;
end;

class function TVdxQwen35Model.SupportedArchitectures(): TArray<string>;
begin
  Result := ['qwen35'];
end;

function TVdxQwen35Model.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
var P: string; NativeContext, TotalLayers, MTP: UInt32;
begin
  inherited LoadModelConfig(AReader, AMaxContext);
  P := 'qwen35.';
  TotalLayers := AReader.GetMetadataUInt32(P+'block_count');
  MTP := AReader.GetMetadataUInt32(P+'nextn_predict_layers',0);
  if (TotalLayers <> 24+MTP) or (MTP>1) then
    raise ENotSupportedException.Create('Native qwen35 currently supports the 0.8B text model (24 decoder layers)');
  FNumLayers := TotalLayers-MTP;
  FHiddenDim := AReader.GetMetadataUInt32(P+'embedding_length');
  FFFNWidth := AReader.GetMetadataUInt32(P+'feed_forward_length');
  FNumQHeads := AReader.GetMetadataUInt32(P+'attention.head_count');
  FNumKVHeads := AReader.GetMetadataUInt32(P+'attention.head_count_kv');
  FHeadDim := AReader.GetMetadataUInt32(P+'attention.key_length');
  if (FHiddenDim<>1024) or (FFFNWidth<>3584) or (FNumQHeads<>8) or
    (FNumKVHeads<>2) or (FHeadDim<>256) or
    (AReader.GetMetadataUInt32(P+'attention.value_length')<>256) or
    (AReader.GetMetadataUInt32(P+'rope.dimension_count')<>64) or
    (AReader.GetMetadataUInt32(P+'ssm.conv_kernel')<>4) or
    (AReader.GetMetadataUInt32(P+'ssm.state_size')<>128) or
    (AReader.GetMetadataUInt32(P+'ssm.group_count')<>16) or
    (AReader.GetMetadataUInt32(P+'ssm.time_step_rank')<>16) or
    (AReader.GetMetadataUInt32(P+'ssm.inner_size')<>2048) or
    (AReader.GetMetadataUInt32(P+'full_attention_interval',4)<>4) then
    raise ENotSupportedException.Create('Unsupported qwen35 dimensions: expected Qwen3.5-0.8B');
  NativeContext := AReader.GetMetadataUInt32(P+'context_length');
  if AMaxContext<=0 then FMaxSeqLen := Min(NativeContext,UInt32(4096))
  else FMaxSeqLen := Min(NativeContext,UInt32(AMaxContext));
  if (FMaxSeqLen=0) or (FMaxSeqLen>65535) then
    raise ERangeError.Create('Native Qwen35 context must be 1..65535');
  FParams := Default(TVdxQwen35Push);
  FParams.MaxSeq := FMaxSeqLen; FParams.Heads := FNumQHeads;
  FParams.KVHeads := FNumKVHeads; FParams.HeadDim := FHeadDim;
  FParams.Eps := AReader.GetMetadataFloat32(P+'attention.layer_norm_rms_epsilon');
  FParams.Theta := AReader.GetMetadataFloat32(P+'rope.freq_base');
  if (FParams.Eps<=0) or (FParams.Theta<=1) then raise EConvertError.Create('Invalid Qwen35 norm/RoPE metadata');
  FBatchCount := 1; // recurrent state advances strictly in token order
  Status('Native Qwen3.5-0.8B: 24 layers (18 SSM, 6 attention), context=%d; MTP skipped=%d',[FMaxSeqLen,MTP]);
  Result := True;
end;

function TVdxQwen35Model.NewBuffer(const ABytes: UInt64; const AHost: Boolean): TVdxGpuBuffer;
var Flags: VkFlags;
begin
  if AHost then Flags := VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  else Flags := VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  Result := FCompute.CreateGpuBuffer((ABytes+3) and not UInt64(3),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT, Flags);
  FOwned.Add(Result);
  if FErrors.HasErrors() or (Result.Memory = VK_NULL_HANDLE) then
    raise EOutOfMemory.CreateFmt('Cannot allocate %d bytes on Vulkan device', [ABytes]);
end;

function TVdxQwen35Model.InitSubsystems(): Boolean;
var Stream: TResourceStream;
  function Scratch(const N: UInt64; const Host: Boolean = False): TVdxGpuBuffer;
  begin Result := NewBuffer(N,Host); Inc(FScratchBytes,Result.Size); end;
begin
  Result := False;
  if not FTokenizer.LoadFromGGUF(FReader) then Exit;
  FVocabSize := FTokenizer.GetVocabSize();
  FCompute.Init();
  if FErrors.HasErrors() then Exit;
  Stream := TResourceStream.Create(HInstance,'QWEN35_DENSE',RT_RCDATA);
  try FShader := FCompute.CreateShaderModule(Stream.Memory,Stream.Size);
  finally Stream.Free(); end;
  FLayout := FCompute.CreateStorageDescriptorSetLayout(5);
  FKernel := FCompute.CreateComputePipelineWithPush(FShader,'main',FLayout,SizeOf(FParams));
  if FErrors.HasErrors() then raise Exception.Create(FErrors.ToString());
  FDummy := Scratch(4,True);
  FResidualMat := Scratch(UInt64(FHiddenDim)*4);
  FResidualGpu := Scratch(UInt64(FHiddenDim)*4);
  FWork := Scratch(UInt64(FHiddenDim)*4);
  FQ := Scratch(6144*4); FConv := Scratch(6144*4);
  FQGate := Scratch(4096*4);
  FK := Scratch(512*4); FV := Scratch(512*4);
  FZ := Scratch(2048*4); FAlpha := Scratch(16*4);
  FBeta := Scratch(16*4); FDecay := Scratch(16*4);
  FAttended := Scratch(2048*4); FProjected := Scratch(UInt64(FHiddenDim)*4);
  FGate := Scratch(UInt64(FFFNWidth)*4); FUp := Scratch(UInt64(FFFNWidth)*4);
  FScores := Scratch(UInt64(FNumQHeads)*FMaxSeqLen*4);
  FTokenIds := Scratch(4,True);
  FLogitsBuf := Scratch(UInt64(FVocabSize)*4,True);
  FLogitsVBuf := TVdxVirtualBuffer<Single>.Create(); FLogitsVBuf.Allocate(FVocabSize);
  Result := not FErrors.HasErrors();
end;

function TVdxQwen35Model.ReadTensor(const AName: string;
  const AWidth, ARows: UInt32; const ANorm: Boolean): TVdxQwen35Tensor;
var Info: TVdxGGUFTensorInfo; N: UInt64; Staging: TVdxGpuBuffer; Data: Pointer;
begin
  Result := Default(TVdxQwen35Tensor);
  if not FReader.GetTensorInfo(AName, Info) then raise EConvertError.Create('Missing tensor: '+AName);
  if (Length(Info.Dimensions) < 1) or (Info.Dimensions[0] <> AWidth) then
    raise EConvertError.Create('Incorrect tensor width: '+AName);
  if ANorm then
  begin
    if (Info.NumDimensions <> 1) or (Info.TensorType <> gtF32) then
      raise EConvertError.Create('Norm must be a one-dimensional F32 tensor: '+AName);
  end
  else if (Info.NumDimensions <> 2) or (Info.Dimensions[1] <> ARows) then
    raise EConvertError.Create('Incorrect tensor shape: '+AName);
  if not (Info.TensorType in [gtF32,gtF16,gtQ4_0,gtQ4_1,gtQ8_0,gtQ3_K,gtQ4_K,gtQ6_K]) then
    raise ENotSupportedException.CreateFmt('Unsupported tensor %s: %s',[AName,VdxGGMLTypeName(Info.TensorType)]);
  N := VdxGGMLTensorBytes(Info.TensorType,AWidth,ARows);
  Data := FReader.GetTensorDataPtr(AName,N);
  if (N=0) or (Data=nil) then raise EConvertError.Create('Invalid tensor data: '+AName);
  Result.Buffer := NewBuffer(N);
  Result.Kind := Info.TensorType; Result.Width := AWidth; Result.Rows := ARows;
  Inc(FWeightBytes,Result.Buffer.AllocationBytes);
  Staging := FCompute.CreateGpuBuffer(Result.Buffer.Size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    FCompute.UploadToBuffer(Staging,Data,N);
    FCompute.CopyBuffer(Staging,Result.Buffer,N);
  finally FCompute.DestroyGpuBuffer(Staging); end;
end;

function TVdxQwen35Model.LoadWeights(): Boolean;
var I: Integer; P: string; ZeroPush: TVdxQwen35Push;
begin
  FEmbedding := ReadTensor('token_embd.weight',FHiddenDim,FVocabSize);
  FEmbedType := FEmbedding.Kind;
  if FReader.HasTensor('output.weight') then FOutput := ReadTensor('output.weight',FHiddenDim,FVocabSize)
  else FOutput := FEmbedding;
  FFinalNorm := ReadTensor('output_norm.weight',FHiddenDim,1,True);
  SetLength(FLayers,FNumLayers);
  for I := 0 to Integer(FNumLayers)-1 do
  begin
    Status('Uploading Qwen35 layer %d/%d',[I+1,FNumLayers]);
    P := Format('blk.%d.',[I]);
    FLayers[I].AttnNorm := ReadTensor(P+'attn_norm.weight',FHiddenDim,1,True);
    FLayers[I].FFNNorm := ReadTensor(P+'post_attention_norm.weight',FHiddenDim,1,True);
    FLayers[I].Gate := ReadTensor(P+'ffn_gate.weight',FHiddenDim,FFFNWidth);
    FLayers[I].Up := ReadTensor(P+'ffn_up.weight',FHiddenDim,FFFNWidth);
    FLayers[I].Down := ReadTensor(P+'ffn_down.weight',FFFNWidth,FHiddenDim);
    // Both cache slots use a uniform padded size for the existing snapshot format.
    FLayers[I].KCache := NewBuffer(CacheBytesPerLayer());
    FLayers[I].VCache := NewBuffer(CacheBytesPerLayer());
    Inc(FCacheBytes,FLayers[I].KCache.AllocationBytes+FLayers[I].VCache.AllocationBytes);
    ZeroPush := FParams; ZeroPush.Op := 18; ZeroPush.InDim := CacheBytesPerLayer() div 4;
    // Initialize padding as well: cache snapshots must never expose uninitialized GPU memory.
    DispatchKernel(ZeroPush,FDummy,FDummy,FDummy,FLayers[I].KCache,(ZeroPush.InDim+255) div 256);
    DispatchKernel(ZeroPush,FDummy,FDummy,FDummy,FLayers[I].VCache,(ZeroPush.InDim+255) div 256);
    if (I+1) mod 4=0 then
    begin
      FLayers[I].Q := ReadTensor(P+'attn_q.weight',FHiddenDim,4096);
      FLayers[I].K := ReadTensor(P+'attn_k.weight',FHiddenDim,512);
      FLayers[I].V := ReadTensor(P+'attn_v.weight',FHiddenDim,512);
      FLayers[I].O := ReadTensor(P+'attn_output.weight',2048,FHiddenDim);
      FLayers[I].QNorm := ReadTensor(P+'attn_q_norm.weight',256,1,True);
      FLayers[I].KNorm := ReadTensor(P+'attn_k_norm.weight',256,1,True);
    end
    else
    begin
      FLayers[I].Q := ReadTensor(P+'attn_qkv.weight',FHiddenDim,6144);
      FLayers[I].Z := ReadTensor(P+'attn_gate.weight',FHiddenDim,2048);
      FLayers[I].O := ReadTensor(P+'ssm_out.weight',2048,FHiddenDim);
      FLayers[I].Conv := ReadTensor(P+'ssm_conv1d.weight',4,6144);
      if FLayers[I].Conv.Kind<>gtF32 then raise ENotSupportedException.Create('SSM convolution requires F32 weights');
      FLayers[I].Alpha := ReadTensor(P+'ssm_alpha.weight',FHiddenDim,16);
      FLayers[I].Beta := ReadTensor(P+'ssm_beta.weight',FHiddenDim,16);
      FLayers[I].Decay := ReadTensor(P+'ssm_a',16,1,True);
      FLayers[I].DT := ReadTensor(P+'ssm_dt.bias',16,1,True);
      FLayers[I].SNorm := ReadTensor(P+'ssm_norm.weight',128,1,True);
    end;
  end;
  FWeightType := FLayers[0].Q.Kind;
  Result := not FErrors.HasErrors();
end;

procedure TVdxQwen35Model.DispatchKernel(const APush: TVdxQwen35Push;
  const A, B, C, D: TVdxGpuBuffer; const AX, AY, AZ: UInt32; const AState: PVdxQwen35Buffer);
var Pool: VkDescriptorPool; Desc: VkDescriptorSet; StateBuffer: TVdxGpuBuffer;
begin
  // Each recorded dispatch owns immutable descriptors until EndBatch's fence.
  // Rebinding a descriptor set during command recording changes earlier dispatches too.
  if AState=nil then StateBuffer := FDummy else StateBuffer := AState^;
  Pool := FCompute.CreateDescriptorPoolForStorage(1,5);
  try
    Desc := FCompute.AllocateDescriptorSetForBuffers(Pool,FLayout,[A,B,C,D,StateBuffer]);
    FCompute.DispatchComputeWithPush(FKernel.Pipeline,FKernel.PipelineLayout,Desc,
      @APush,SizeOf(APush),AX,AY,AZ);
    FCompute.BatchBarrier();
    if FErrors.HasErrors() then raise Exception.Create(FErrors.ToString());
  finally FCompute.DestroyDescriptorPoolHandle(Pool); end;
end;

procedure TVdxQwen35Model.MatMul(const W: TVdxQwen35Tensor;
  const X, Y: TVdxGpuBuffer; const N: UInt32);
var P: TVdxQwen35Push;
begin
  P := FParams; P.Op := 2; P.InDim := W.Width; P.OutDim := W.Rows; P.TensorType := Ord(W.Kind); P.Tokens := N;
  DispatchKernel(P,W.Buffer,X,FDummy,Y,Min(W.Rows,UInt32(65535)),N,(W.Rows+65534) div 65535);
end;

procedure TVdxQwen35Model.Norm(const X: TVdxGpuBuffer; const W: TVdxQwen35Tensor;
  const Y: TVdxGpuBuffer; const N: UInt32);
var P: TVdxQwen35Push;
begin
  P := FParams; P.Op := 1; P.InDim := W.Width;
  DispatchKernel(P,X,W.Buffer,FDummy,Y,N);
end;

procedure TVdxQwen35Model.Layer(const ALayer: Integer; const N, Position: UInt32);
var L: TVdxQwen35Layer; P: TVdxQwen35Push;
begin
  if (N<>1) or (Position>=FMaxSeqLen) then raise ERangeError.Create('Qwen35 requires single-token recurrent steps');
  L := FLayers[ALayer];
  Norm(FResidualMat,L.AttnNorm,FWork,1);
  if (ALayer+1) mod 4=0 then
  begin
    MatMul(L.Q,FWork,FQGate,1); MatMul(L.K,FWork,FK,1); MatMul(L.V,FWork,FV,1);
    P := FParams; P.Op := 15;
    DispatchKernel(P,FQGate,L.QNorm.Buffer,FDummy,FQ,FNumQHeads);
    Norm(FK,L.KNorm,FK,FNumKVHeads);
    P.Op := 16; P.StartPos := Position; P.OutDim := 64;
    DispatchKernel(P,FQ,FDummy,FDummy,FQ,FNumQHeads);
    DispatchKernel(P,FK,FDummy,FDummy,FK,FNumKVHeads);
    P := FParams; P.Op := 4; P.StartPos := Position;
    DispatchKernel(P,FK,FDummy,FDummy,L.KCache,1); DispatchKernel(P,FV,FDummy,FDummy,L.VCache,1);
    P.Op := 5;
    DispatchKernel(P,FQ,L.KCache,FDummy,FScores,Position+1,1,FNumQHeads);
    P.Op := 6; DispatchKernel(P,FScores,FDummy,FDummy,FScores,FNumQHeads);
    P.Op := 7; DispatchKernel(P,FScores,L.VCache,FDummy,FAttended,FNumQHeads);
    P.Op := 17; P.InDim := 2048;
    DispatchKernel(P,FAttended,FQGate,FDummy,FAttended,8);
  end
  else
  begin
    MatMul(L.Q,FWork,FQ,1); MatMul(L.Z,FWork,FZ,1);
    MatMul(L.Alpha,FWork,FAlpha,1); MatMul(L.Beta,FWork,FBeta,1);
    P := FParams; P.Op := 10; P.InDim := 6144; P.StartPos := Position;
    DispatchKernel(P,FQ,L.Conv.Buffer,FDummy,FConv,24,1,1,@L.KCache);
    P.Op := 11; P.InDim := 128;
    DispatchKernel(P,FConv,FDummy,FDummy,FConv,32);
    P.Op := 12; P.InDim := 16;
    DispatchKernel(P,FAlpha,L.DT.Buffer,L.Decay.Buffer,FDecay,1);
    P.Op := 13; P.InDim := 128; P.Heads := 16;
    DispatchKernel(P,FConv,FDecay,FBeta,FAttended,128,16,1,@L.VCache);
    P.Op := 14;
    DispatchKernel(P,FAttended,L.SNorm.Buffer,FZ,FAttended,16);
  end;
  MatMul(L.O,FAttended,FProjected,1);
  P := FParams; P.Op := 9; P.InDim := FHiddenDim; P.Tokens := 1;
  DispatchKernel(P,FResidualMat,FProjected,FDummy,FResidualMat,(FHiddenDim+255) div 256);
  Norm(FResidualMat,L.FFNNorm,FWork,1);
  MatMul(L.Gate,FWork,FGate,1); MatMul(L.Up,FWork,FUp,1);
  P.Op := 8; P.InDim := FFFNWidth;
  DispatchKernel(P,FGate,FUp,FDummy,FGate,(FFFNWidth+255) div 256);
  MatMul(L.Down,FGate,FProjected,1);
  P.Op := 9; P.InDim := FHiddenDim;
  DispatchKernel(P,FResidualMat,FProjected,FDummy,FResidualMat,(FHiddenDim+255) div 256);
end;

procedure TVdxQwen35Model.RunLayerForward(const ALayer, APosition: Integer);
begin
  Layer(ALayer,1,APosition);
  if ALayer = Integer(FNumLayers)-1 then
    FCompute.CopyBuffer(FResidualMat,FResidualGpu,UInt64(FHiddenDim)*4);
end;

procedure TVdxQwen35Model.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens, AStartPos: UInt32; const ABidirectional: Boolean);
begin
  if ABidirectional then raise ENotSupportedException.Create('Qwen35 is a causal text model');
  Layer(ALayer,ANumTokens,AStartPos);
end;

procedure TVdxQwen35Model.EmbedToken(const ATokenId: Integer);
begin
  EmbedTokensBatch(TArray<Integer>.Create(ATokenId),1,FResidualMat);
end;

procedure TVdxQwen35Model.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
  const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
var P: TVdxQwen35Push; I: Integer;
begin
  if (ANumTokens<=0) or (ANumTokens>Integer(FBatchCount)) or (ANumTokens>Length(ATokenIds)) then
    raise ERangeError.Create('Invalid Qwen35 embedding batch');
  for I := 0 to ANumTokens-1 do
    if (ATokenIds[I]<0) or (ATokenIds[I]>=FVocabSize) then raise ERangeError.Create('Invalid token');
  FCompute.UploadToBuffer(FTokenIds,@ATokenIds[0],UInt64(ANumTokens)*4);
  P := FParams; P.Op := 0; P.InDim := FHiddenDim; P.TensorType := Ord(FEmbedding.Kind);
  DispatchKernel(P,FEmbedding.Buffer,FTokenIds,FDummy,AOutputBuf,ANumTokens);
end;

procedure TVdxQwen35Model.SeedResidualFromBatchLast(const ANumTokens: UInt32);
begin
  FCompute.CopyBufferRegion(FResidualMat,UInt64(ANumTokens-1)*FHiddenDim*4,FResidualGpu,0,UInt64(FHiddenDim)*4);
end;

procedure TVdxQwen35Model.UnembedToLogits(const AOutLogits: TVdxGpuBuffer);
begin
  FCompute.BeginBatch();
  try
    Norm(FResidualGpu,FFinalNorm,FWork,1);
    MatMul(FOutput,FWork,AOutLogits,1);
  finally FCompute.EndBatch(); end;
end;

function TVdxQwen35Model.FormatPrompt(const APrompt: string): string;
begin
  Result := '<|im_start|>user'+#10+APrompt+'<|im_end|>'+#10+
    '<|im_start|>assistant'+#10+'<think>'+#10+'</think>'+#10+#10;
end;

function TVdxQwen35Model.GetStopTokenStrings(): TArray<string>;
begin Result := ['<|im_end|>','<|endoftext|>']; end;
function TVdxQwen35Model.PrefillBatchSize(): Integer;
begin Result := FBatchCount; end;
function TVdxQwen35Model.CacheFormat(): UInt32;
begin Result := 2; end;
function TVdxQwen35Model.CacheBytesPerLayer(): UInt64;
begin Result := Max(UInt64(FMaxSeqLen)*FNumKVHeads*FHeadDim*4,UInt64(16*128*128*4)); end;
function TVdxQwen35Model.CacheBuffer(const ALayer: Integer; const AKey: Boolean): TVdxGpuBuffer;
begin
  if AKey then Result := FLayers[ALayer].KCache else Result := FLayers[ALayer].VCache;
end;
function TVdxQwen35Model.AllocatedBytes(var AWeights, ACache, AScratch: UInt64): Boolean;
begin
  AWeights := FWeightBytes; ACache := FCacheBytes;
  AScratch := FCompute.AllocatedBytes - AWeights - ACache; Result := True;
end;

procedure TVdxQwen35Model.FreeWeights();
var B: TVdxGpuBuffer; I: Integer;
begin
  if FOwned <> nil then
  begin
    for I := 0 to FOwned.Count-1 do
    begin
      B := FOwned[I];
      // DestroyGpuBuffer zeroes its var parameter; this is a copy of the owned record.
      FCompute.DestroyGpuBuffer(B);
    end;
    FOwned.Clear();
  end;
  FResidualMat := Default(TVdxGpuBuffer); FResidualGpu := Default(TVdxGpuBuffer);
  FLogitsBuf := Default(TVdxGpuBuffer);
  if FKernel.Pipeline <> VK_NULL_HANDLE then FCompute.DestroyComputePipelineBundle(FKernel);
  if FLayout <> VK_NULL_HANDLE then FCompute.DestroyDescriptorSetLayoutHandle(FLayout);
  if FShader <> VK_NULL_HANDLE then FCompute.DestroyShaderModuleHandle(FShader);
  FKernel := Default(TVdxComputePipelineBundle); FLayout := VK_NULL_HANDLE; FShader := VK_NULL_HANDLE;
  FLayers := nil;
end;

initialization
  TVdxModelRegistry.RegisterClass(TVdxQwen35Model);
end.
