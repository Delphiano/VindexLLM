unit VindexLLM.Model.Ministral3;

interface

uses
  System.SysUtils, System.Generics.Collections,
  VindexLLM.Model, VindexLLM.GGUFReader, VindexLLM.Compute, VindexLLM.Vulkan;

type
  // Fields match the GLSL push-constant block (72 bytes, all scalar 32-bit).
  TVdxMinistralPush = record
    Op, InDim, OutDim, TensorType: UInt32;
    Tokens, StartPos, MaxSeq, Heads, KVHeads, HeadDim: UInt32;
    Eps, Theta, FreqScale, CorrLow, CorrHigh, Magnitude, TempScale: Single;
    OrigContext: UInt32;
  end;

  TVdxMinistralTensor = record
    Buffer: TVdxGpuBuffer;
    Kind: TVdxGGMLType;
    Width, Rows: UInt32;
  end;

  TVdxMinistralLayer = record
    Q, K, V, O, Gate, Up, Down, AttnNorm, FFNNorm: TVdxMinistralTensor;
    KCache, VCache: TVdxGpuBuffer;
  end;

  TVdxMinistral3Model = class(TVdxModel)
  private
    FLayers: TArray<TVdxMinistralLayer>;
    FEmbedding, FOutput, FFinalNorm: TVdxMinistralTensor;
    FOwned: TList<TVdxGpuBuffer>;
    FLayout: VkDescriptorSetLayout;
    FShader: VkShaderModule;
    FKernel: TVdxComputePipelineBundle;
    FParams: TVdxMinistralPush;
    FQ, FK, FV, FWork, FAttended, FProjected, FGate, FUp, FScores: TVdxGpuBuffer;
    FTokenIds, FDummy, FRotaryFreqs: TVdxGpuBuffer;
    FWeightBytes, FCacheBytes, FScratchBytes: UInt64;
    FBatchCount: UInt32;
    function NewBuffer(const ABytes: UInt64; const AHost: Boolean = False): TVdxGpuBuffer;
    function ReadTensor(const AName: string; const AWidth, ARows: UInt32;
      const ANorm: Boolean = False): TVdxMinistralTensor;
    procedure DispatchKernel(const APush: TVdxMinistralPush;
      const A, B, C, D: TVdxGpuBuffer; const AX: UInt32;
      const AY: UInt32 = 1; const AZ: UInt32 = 1);
    procedure MatMul(const W: TVdxMinistralTensor; const X, Y: TVdxGpuBuffer; const N: UInt32);
    procedure Norm(const X: TVdxGpuBuffer; const W: TVdxMinistralTensor;
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

{$R ministral\Ministral.res}

constructor TVdxMinistral3Model.Create();
begin
  inherited;
  FOwned := TList<TVdxGpuBuffer>.Create();
end;

destructor TVdxMinistral3Model.Destroy();
begin
  FreeWeights();
  FreeAndNil(FOwned);
  inherited;
end;

class function TVdxMinistral3Model.SupportedArchitectures(): TArray<string>;
begin
  Result := ['mistral3'];
end;

function TVdxMinistral3Model.LoadModelConfig(const AReader: TVdxGGUFReader;
  const AMaxContext: Integer): Boolean;
var P: string; NativeContext: UInt32; Factor, LogMul: Single;
begin
  inherited LoadModelConfig(AReader, AMaxContext);
  P := FArchitecture + '.';
  FNumLayers := AReader.GetMetadataUInt32(P+'block_count');
  FHiddenDim := AReader.GetMetadataUInt32(P+'embedding_length');
  FFFNWidth := AReader.GetMetadataUInt32(P+'feed_forward_length');
  FNumQHeads := AReader.GetMetadataUInt32(P+'attention.head_count');
  FNumKVHeads := AReader.GetMetadataUInt32(P+'attention.head_count_kv');
  FHeadDim := AReader.GetMetadataUInt32(P+'attention.key_length');
  // Deliberately validate the target family instead of guessing missing dimensions.
  if (FNumLayers <> 26) or (FHiddenDim <> 3072) or (FFFNWidth <> 9216) or
     (FNumQHeads <> 32) or (FNumKVHeads <> 8) or (FHeadDim <> 128) or
     (AReader.GetMetadataUInt32(P+'attention.value_length') <> FHeadDim) or
     (AReader.GetMetadataUInt32(P+'rope.dimension_count') <> FHeadDim) then
    raise ENotSupportedException.Create('This native mistral3 implementation supports Ministral 3 3B (26 layers)');
  if AReader.GetMetadataString(P+'rope.scaling.type') <> 'yarn' then
    raise ENotSupportedException.Create('Ministral requires YaRN RoPE metadata');
  NativeContext := AReader.GetMetadataUInt32(P+'context_length');
  if AMaxContext <= 0 then FMaxSeqLen := Min(NativeContext, UInt32(4096))
  else FMaxSeqLen := Min(NativeContext, UInt32(AMaxContext));
  if (FMaxSeqLen = 0) or (FMaxSeqLen > 65535) then
    raise ENotSupportedException.Create('Native Ministral context must be 1..65535; start with 4096');
  FParams := Default(TVdxMinistralPush);
  FParams.MaxSeq := FMaxSeqLen;
  FParams.Heads := FNumQHeads;
  FParams.KVHeads := FNumKVHeads;
  FParams.HeadDim := FHeadDim;
  FParams.Eps := AReader.GetMetadataFloat32(P+'attention.layer_norm_rms_epsilon');
  FParams.Theta := AReader.GetMetadataFloat32(P+'rope.freq_base');
  FParams.OrigContext := AReader.GetMetadataUInt32(P+'rope.scaling.original_context_length');
  FParams.TempScale := AReader.GetMetadataFloat32(P+'attention.temperature_scale');
  Factor := AReader.GetMetadataFloat32(P+'rope.scaling.factor');
  if (Factor < 1) or (FParams.Theta <= 1) or (FParams.OrigContext = 0) or (FParams.Eps <= 0) then
    raise EConvertError.Create('Invalid Ministral normalization or YaRN parameters');
  FParams.FreqScale := 1/Factor;
  FParams.CorrLow := Max(0, Floor(FHeadDim * Ln(FParams.OrigContext /
    (AReader.GetMetadataFloat32(P+'rope.scaling.yarn_beta_fast',32)*2*Pi)) / (2*Ln(FParams.Theta))));
  FParams.CorrHigh := Min(Integer(FHeadDim)-1, Ceil(FHeadDim * Ln(FParams.OrigContext /
    (AReader.GetMetadataFloat32(P+'rope.scaling.yarn_beta_slow',1)*2*Pi)) / (2*Ln(FParams.Theta))));
  LogMul := AReader.GetMetadataFloat32(P+'rope.scaling.yarn_log_multiplier');
  FParams.Magnitude := 1+0.1*Ln(Factor);
  if LogMul <> 0 then FParams.Magnitude := FParams.Magnitude/(1+0.1*LogMul*Ln(Factor));
  FBatchCount := Min(UInt32(32), FMaxSeqLen);
  Status('Native Ministral 3 3B: context=%d, prefill block=%d, F32 KV cache', [FMaxSeqLen, FBatchCount]);
  Result := True;
end;

function TVdxMinistral3Model.NewBuffer(const ABytes: UInt64; const AHost: Boolean): TVdxGpuBuffer;
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

function TVdxMinistral3Model.InitSubsystems(): Boolean;
var Stream: TResourceStream; Freqs: TArray<Single>; I: Integer; Freq, Ramp: Double;
  function Scratch(const N: UInt64; const Host: Boolean = False): TVdxGpuBuffer;
  begin Result := NewBuffer(N,Host); Inc(FScratchBytes,Result.Size); end;
begin
  Result := False;
  // Load and validate vocabulary before allocating GPU weights.
  if not FTokenizer.LoadFromGGUF(FReader) then Exit;
  FVocabSize := FTokenizer.GetVocabSize();
  if FVocabSize <> 131072 then raise EConvertError.Create('Unexpected Ministral vocabulary size');
  FCompute.Init();
  if FErrors.HasErrors() then Exit;
  Stream := TResourceStream.Create(HInstance,'MINISTRAL_DENSE',RT_RCDATA);
  try FShader := FCompute.CreateShaderModule(Stream.Memory,Stream.Size);
  finally Stream.Free(); end;
  FLayout := FCompute.CreateStorageDescriptorSetLayout(4);
  FKernel := FCompute.CreateComputePipelineWithPush(FShader,'main',FLayout,SizeOf(FParams));
  if FErrors.HasErrors() then raise Exception.Create(FErrors.ToString());
  FDummy := Scratch(4,True);
  FRotaryFreqs := Scratch(UInt64(FHeadDim)*4,True);
  SetLength(Freqs,FHeadDim);
  for I := 0 to Integer(FHeadDim div 2)-1 do
  begin
    Ramp := 1-Max(0.0,Min(1.0,(I-FParams.CorrLow)/Max(0.001,FParams.CorrHigh-FParams.CorrLow)));
    Freq := Power(Double(FParams.Theta),-2.0*I/FHeadDim)*(FParams.FreqScale*(1-Ramp)+Ramp);
    Freqs[2*I] := Freq;
    Freqs[2*I+1] := Freq-Double(Freqs[2*I]);
  end;
  FCompute.UploadToBuffer(FRotaryFreqs,@Freqs[0],UInt64(FHeadDim)*4);
  FResidualMat := Scratch(UInt64(FBatchCount)*FHiddenDim*4);
  FResidualGpu := Scratch(UInt64(FHiddenDim)*4);
  FWork := Scratch(UInt64(FBatchCount)*FHiddenDim*4);
  FQ := Scratch(UInt64(FBatchCount)*FNumQHeads*FHeadDim*4);
  FK := Scratch(UInt64(FBatchCount)*FNumKVHeads*FHeadDim*4);
  FV := Scratch(UInt64(FBatchCount)*FNumKVHeads*FHeadDim*4);
  FAttended := Scratch(UInt64(FBatchCount)*FNumQHeads*FHeadDim*4);
  FProjected := Scratch(UInt64(FBatchCount)*FHiddenDim*4);
  FGate := Scratch(UInt64(FBatchCount)*FFFNWidth*4);
  FUp := Scratch(UInt64(FBatchCount)*FFFNWidth*4);
  FScores := Scratch(UInt64(FBatchCount)*FNumQHeads*FMaxSeqLen*4);
  FTokenIds := Scratch(UInt64(FBatchCount)*4,True);
  FLogitsBuf := Scratch(UInt64(FVocabSize)*4,True);
  FLogitsVBuf := TVdxVirtualBuffer<Single>.Create();
  FLogitsVBuf.Allocate(FVocabSize);
  Result := not FErrors.HasErrors();
end;

function TVdxMinistral3Model.ReadTensor(const AName: string;
  const AWidth, ARows: UInt32; const ANorm: Boolean): TVdxMinistralTensor;
var Info: TVdxGGUFTensorInfo; N: UInt64; Staging: TVdxGpuBuffer; Data: Pointer;
begin
  Result := Default(TVdxMinistralTensor);
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
  if not (Info.TensorType in [gtF32,gtF16,gtQ4_0,gtQ4_1,gtQ8_0,gtQ3_K,gtQ6_K]) then
    raise ENotSupportedException.CreateFmt('Unsupported tensor %s: %s',[AName,VdxGGMLTypeName(Info.TensorType)]);
  N := VdxGGMLTensorBytes(Info.TensorType,AWidth,ARows);
  Data := FReader.GetTensorDataPtr(AName,N);
  if (N=0) or (Data=nil) then raise EConvertError.Create('Invalid tensor data: '+AName);
  Result.Buffer := NewBuffer(N);
  Result.Kind := Info.TensorType; Result.Width := AWidth; Result.Rows := ARows;
  Inc(FWeightBytes,Result.Buffer.Size);
  Staging := FCompute.CreateGpuBuffer(Result.Buffer.Size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    FCompute.UploadToBuffer(Staging,Data,N);
    FCompute.CopyBuffer(Staging,Result.Buffer,N);
  finally FCompute.DestroyGpuBuffer(Staging); end;
end;

function TVdxMinistral3Model.LoadWeights(): Boolean;
var I: Integer; P: string; Info: TVdxGGUFTensorInfo;
begin
  // Refuse extra executable tensors rather than silently ignoring biases/adapters.
  for Info in FReader.GetTensorList() do
    if not Info.TensorName.EndsWith('.weight') then
      raise ENotSupportedException.Create('Unexpected model tensor: '+Info.TensorName);
  FEmbedding := ReadTensor('token_embd.weight',FHiddenDim,FVocabSize);
  FEmbedType := FEmbedding.Kind;
  if FReader.HasTensor('output.weight') then FOutput := ReadTensor('output.weight',FHiddenDim,FVocabSize)
  else FOutput := FEmbedding;
  FFinalNorm := ReadTensor('output_norm.weight',FHiddenDim,1,True);
  SetLength(FLayers,FNumLayers);
  for I := 0 to Integer(FNumLayers)-1 do
  begin
    Status('Uploading Ministral layer %d/%d',[I+1,FNumLayers]);
    P := Format('blk.%d.',[I]);
    FLayers[I].Q := ReadTensor(P+'attn_q.weight',FHiddenDim,FNumQHeads*FHeadDim);
    FLayers[I].K := ReadTensor(P+'attn_k.weight',FHiddenDim,FNumKVHeads*FHeadDim);
    FLayers[I].V := ReadTensor(P+'attn_v.weight',FHiddenDim,FNumKVHeads*FHeadDim);
    FLayers[I].O := ReadTensor(P+'attn_output.weight',FNumQHeads*FHeadDim,FHiddenDim);
    FLayers[I].Gate := ReadTensor(P+'ffn_gate.weight',FHiddenDim,FFFNWidth);
    FLayers[I].Up := ReadTensor(P+'ffn_up.weight',FHiddenDim,FFFNWidth);
    FLayers[I].Down := ReadTensor(P+'ffn_down.weight',FFFNWidth,FHiddenDim);
    FLayers[I].AttnNorm := ReadTensor(P+'attn_norm.weight',FHiddenDim,1,True);
    FLayers[I].FFNNorm := ReadTensor(P+'ffn_norm.weight',FHiddenDim,1,True);
    FLayers[I].KCache := NewBuffer(CacheBytesPerLayer());
    FLayers[I].VCache := NewBuffer(CacheBytesPerLayer());
    Inc(FCacheBytes,2*CacheBytesPerLayer());
  end;
  FWeightType := FLayers[0].Q.Kind;
  Result := not FErrors.HasErrors();
end;

procedure TVdxMinistral3Model.DispatchKernel(const APush: TVdxMinistralPush;
  const A, B, C, D: TVdxGpuBuffer; const AX, AY, AZ: UInt32);
var Pool: VkDescriptorPool; Desc: VkDescriptorSet;
begin
  // Each recorded dispatch owns immutable descriptors until EndBatch's fence.
  // Rebinding a descriptor set during command recording changes earlier dispatches too.
  Pool := FCompute.CreateDescriptorPoolForStorage(1,4);
  try
    Desc := FCompute.AllocateDescriptorSetForBuffers(Pool,FLayout,[A,B,C,D]);
    FCompute.DispatchComputeWithPush(FKernel.Pipeline,FKernel.PipelineLayout,Desc,
      @APush,SizeOf(APush),AX,AY,AZ);
    FCompute.BatchBarrier();
    if FErrors.HasErrors() then raise Exception.Create(FErrors.ToString());
  finally FCompute.DestroyDescriptorPoolHandle(Pool); end;
end;

procedure TVdxMinistral3Model.MatMul(const W: TVdxMinistralTensor;
  const X, Y: TVdxGpuBuffer; const N: UInt32);
var P: TVdxMinistralPush;
begin
  P := FParams; P.Op := 2; P.InDim := W.Width; P.OutDim := W.Rows; P.TensorType := Ord(W.Kind); P.Tokens := N;
  DispatchKernel(P,W.Buffer,X,FDummy,Y,Min(W.Rows,UInt32(65535)),N,(W.Rows+65534) div 65535);
end;

procedure TVdxMinistral3Model.Norm(const X: TVdxGpuBuffer; const W: TVdxMinistralTensor;
  const Y: TVdxGpuBuffer; const N: UInt32);
var P: TVdxMinistralPush;
begin
  P := FParams; P.Op := 1; P.InDim := W.Width;
  DispatchKernel(P,X,W.Buffer,FDummy,Y,N);
end;

procedure TVdxMinistral3Model.Layer(const ALayer: Integer; const N, Position: UInt32);
var L: TVdxMinistralLayer; P: TVdxMinistralPush;
begin
  if (N=0) or (N>FBatchCount) or (UInt64(Position)+N>FMaxSeqLen) then
    raise ERangeError.Create('Ministral batch exceeds allocated context');
  L := FLayers[ALayer];
  Norm(FResidualMat,L.AttnNorm,FWork,N);
  MatMul(L.Q,FWork,FQ,N); MatMul(L.K,FWork,FK,N); MatMul(L.V,FWork,FV,N);
  P := FParams; P.Op := 3; P.Tokens := N; P.StartPos := Position; P.TensorType := 1;
  DispatchKernel(P,FQ,FRotaryFreqs,FDummy,FQ,FNumQHeads,N);
  P.Heads := FNumKVHeads; P.TensorType := 0;
  DispatchKernel(P,FK,FRotaryFreqs,FDummy,FK,FNumKVHeads,N);
  P := FParams; P.Op := 4; P.StartPos := Position;
  DispatchKernel(P,FK,FDummy,FDummy,L.KCache,N); DispatchKernel(P,FV,FDummy,FDummy,L.VCache,N);
  P := FParams; P.Op := 5; P.StartPos := Position;
  DispatchKernel(P,FQ,L.KCache,FDummy,FScores,Position+N,N,FNumQHeads);
  P.Op := 6; DispatchKernel(P,FScores,FDummy,FDummy,FScores,FNumQHeads,N);
  P.Op := 7; DispatchKernel(P,FScores,L.VCache,FDummy,FAttended,FNumQHeads,N);
  MatMul(L.O,FAttended,FProjected,N);
  P := FParams; P.Op := 9; P.InDim := FHiddenDim; P.Tokens := N;
  DispatchKernel(P,FResidualMat,FProjected,FDummy,FResidualMat,(N*FHiddenDim+255) div 256);
  Norm(FResidualMat,L.FFNNorm,FWork,N);
  MatMul(L.Gate,FWork,FGate,N); MatMul(L.Up,FWork,FUp,N);
  P.Op := 8; P.InDim := FFFNWidth;
  DispatchKernel(P,FGate,FUp,FDummy,FGate,(N*FFFNWidth+255) div 256);
  MatMul(L.Down,FGate,FProjected,N);
  P.Op := 9; P.InDim := FHiddenDim;
  DispatchKernel(P,FResidualMat,FProjected,FDummy,FResidualMat,(N*FHiddenDim+255) div 256);
end;

procedure TVdxMinistral3Model.RunLayerForward(const ALayer, APosition: Integer);
begin
  Layer(ALayer,1,APosition);
  if ALayer = Integer(FNumLayers)-1 then
    FCompute.CopyBuffer(FResidualMat,FResidualGpu,UInt64(FHiddenDim)*4);
end;

procedure TVdxMinistral3Model.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens, AStartPos: UInt32; const ABidirectional: Boolean);
begin
  if ABidirectional then raise ENotSupportedException.Create('Ministral is a causal text model');
  Layer(ALayer,ANumTokens,AStartPos);
end;

procedure TVdxMinistral3Model.EmbedToken(const ATokenId: Integer);
begin
  EmbedTokensBatch(TArray<Integer>.Create(ATokenId),1,FResidualMat);
end;

procedure TVdxMinistral3Model.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
  const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
var P: TVdxMinistralPush; I: Integer;
begin
  if (ANumTokens<=0) or (ANumTokens>Integer(FBatchCount)) or (ANumTokens>Length(ATokenIds)) then
    raise ERangeError.Create('Invalid Ministral embedding batch');
  for I := 0 to ANumTokens-1 do
    if (ATokenIds[I]<0) or (ATokenIds[I]>=FVocabSize) then raise ERangeError.Create('Invalid token');
  FCompute.UploadToBuffer(FTokenIds,@ATokenIds[0],UInt64(ANumTokens)*4);
  P := FParams; P.Op := 0; P.InDim := FHiddenDim; P.TensorType := Ord(FEmbedding.Kind);
  DispatchKernel(P,FEmbedding.Buffer,FTokenIds,FDummy,AOutputBuf,ANumTokens);
end;

procedure TVdxMinistral3Model.SeedResidualFromBatchLast(const ANumTokens: UInt32);
begin
  FCompute.CopyBufferRegion(FResidualMat,UInt64(ANumTokens-1)*FHiddenDim*4,FResidualGpu,0,UInt64(FHiddenDim)*4);
end;

procedure TVdxMinistral3Model.UnembedToLogits(const AOutLogits: TVdxGpuBuffer);
begin
  FCompute.BeginBatch();
  try
    Norm(FResidualGpu,FFinalNorm,FWork,1);
    MatMul(FOutput,FWork,AOutLogits,1);
  finally FCompute.EndBatch(); end;
end;

function TVdxMinistral3Model.FormatPrompt(const APrompt: string): string;
begin
  Result := '[INST]'+APrompt+'[/INST]';
end;

function TVdxMinistral3Model.GetStopTokenStrings(): TArray<string>;
begin Result := ['</s>']; end;
function TVdxMinistral3Model.PrefillBatchSize(): Integer;
begin Result := FBatchCount; end;
function TVdxMinistral3Model.CacheFormat(): UInt32;
begin Result := 1; end;
function TVdxMinistral3Model.CacheBytesPerLayer(): UInt64;
begin Result := UInt64(FMaxSeqLen)*FNumKVHeads*FHeadDim*4; end;
function TVdxMinistral3Model.CacheBuffer(const ALayer: Integer; const AKey: Boolean): TVdxGpuBuffer;
begin
  if AKey then Result := FLayers[ALayer].KCache else Result := FLayers[ALayer].VCache;
end;
function TVdxMinistral3Model.AllocatedBytes(var AWeights, ACache, AScratch: UInt64): Boolean;
begin
  AWeights := FWeightBytes; ACache := FCacheBytes; AScratch := FScratchBytes; Result := True;
end;

procedure TVdxMinistral3Model.FreeWeights();
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
  TVdxModelRegistry.RegisterClass(TVdxMinistral3Model);
end.
