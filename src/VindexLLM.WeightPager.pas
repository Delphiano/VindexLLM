unit VindexLLM.WeightPager;

interface

uses
  System.SysUtils, System.Generics.Collections, VindexLLM.Compute,
  VindexLLM.GGUFReader, VindexLLM.Vulkan;

type
  TVdxPagedWeight = record
    Name: string;
    Layer, Slot: Integer;
    Data: Pointer;
    Bytes: UInt64;
    Buffer: TVdxGpuBuffer;
  end;

  // Owns resident weights and one reusable set of layer buffers.
  // The reader must outlive this object: Data points into its mapped GGUF.
  TVdxWeightPager = class
  private
    FCompute: TVdxCompute;
    FReader: TVdxGGUFReader;
    FWeights: TList<TVdxPagedWeight>;
    FSlots: TArray<TVdxGpuBuffer>;
    FStaging: TVdxGpuBuffer;
    FResidentLayers, FCurrentLayer: Integer;
    FAllocated: Boolean;
    procedure Upload(const W: TVdxPagedWeight; const B: TVdxGpuBuffer);
  public
    constructor Create(const C: TVdxCompute; const R: TVdxGGUFReader);
    destructor Destroy; override;
    procedure Add(const Name: string; const Layer, Slot: Integer);
    procedure Allocate(const LayerCount: Integer; const BudgetBytes: UInt64);
    function Buffer(const Name: string): TVdxGpuBuffer;
    procedure PrepareLayer(const Layer: Integer);
    function WeightBytes(): UInt64;
    property ResidentLayers: Integer read FResidentLayers;
  end;

implementation

constructor TVdxWeightPager.Create(const C: TVdxCompute; const R: TVdxGGUFReader);
begin
  inherited Create;
  FCompute := C; FReader := R;
  FWeights := TList<TVdxPagedWeight>.Create;
  FCurrentLayer := -1;
end;

destructor TVdxWeightPager.Destroy;
var W: TVdxPagedWeight; B: TVdxGpuBuffer; I: Integer;
begin
  if FWeights <> nil then
    for W in FWeights do
    begin
      B := W.Buffer;
      FCompute.DestroyGpuBuffer(B);
    end;
  for I := 0 to High(FSlots) do FCompute.DestroyGpuBuffer(FSlots[I]);
  FCompute.DestroyGpuBuffer(FStaging);
  FWeights.Free;
  inherited;
end;

procedure TVdxWeightPager.Add(const Name: string; const Layer, Slot: Integer);
var Info: TVdxGGUFTensorInfo; W: TVdxPagedWeight;
begin
  if FAllocated then raise Exception.Create('Weight pager is already allocated');
  if (Layer < 0) or (Slot < 0) then raise Exception.Create('Invalid paging group');
  if not FReader.GetTensorInfo(Name, Info) then
    raise Exception.Create('Missing paged tensor: ' + Name);
  if Length(Info.Dimensions) <> 2 then
    raise Exception.Create('Paged weight must be a matrix: ' + Name);
  W := Default(TVdxPagedWeight);
  W.Name := Name; W.Layer := Layer; W.Slot := Slot;
  W.Bytes := VdxGGMLTensorBytes(Info.TensorType, Info.Dimensions[0], Info.Dimensions[1]);
  if W.Bytes = 0 then raise Exception.Create('Unsupported paged tensor: ' + Name);
  W.Data := FReader.GetTensorDataPtr(Name, W.Bytes);
  if W.Data = nil then raise Exception.Create('Invalid paged tensor data: ' + Name);
  FWeights.Add(W);
end;

procedure TVdxWeightPager.Upload(const W: TVdxPagedWeight; const B: TVdxGpuBuffer);
begin
  // Synchronous copies finish before staging or layer slots are reused.
  FCompute.UploadToBuffer(FStaging, W.Data, W.Bytes);
  FCompute.CopyBuffer(FStaging, B, W.Bytes);
end;

procedure TVdxWeightPager.Allocate(const LayerCount: Integer; const BudgetBytes: UInt64);
var Sizes, Layers: TArray<UInt64>; W: TVdxPagedWeight;
    I: Integer; Total, SlotsTotal, MaxTensor, Available, Used, Budget, OverrideMB: UInt64;
    Setting: string;
begin
  if FAllocated then raise Exception.Create('Weight pager is already allocated');
  FAllocated := True;
  Budget := BudgetBytes;
  Setting := GetEnvironmentVariable('VINDEXLLM_GPU_BUDGET_MB');
  if Setting <> '' then
  begin
    if not TryStrToUInt64(Setting, OverrideMB) or (OverrideMB = 0) or
       (OverrideMB > High(UInt64) div (1024 * 1024)) then
      raise Exception.Create('VINDEXLLM_GPU_BUDGET_MB must be a positive integer in MiB');
    if OverrideMB * (1024 * 1024) < Budget then Budget := OverrideMB * (1024 * 1024);
  end;
  SetLength(Layers, LayerCount);
  Total := 0; MaxTensor := 0;
  for W in FWeights do
  begin
    if W.Layer >= LayerCount then raise Exception.Create('Invalid weight layer');
    if W.Slot >= Length(Sizes) then SetLength(Sizes, W.Slot + 1);
    if W.Bytes > Sizes[W.Slot] then Sizes[W.Slot] := W.Bytes;
    if W.Bytes > MaxTensor then MaxTensor := W.Bytes;
    Inc(Layers[W.Layer], W.Bytes); Inc(Total, W.Bytes);
  end;
  if (Budget <= FCompute.AllocatedBytes) or
     (Budget - FCompute.AllocatedBytes <= MaxTensor) then
    raise Exception.Create('Insufficient GPU budget for weight staging; reduce context length');
  Available := Budget - FCompute.AllocatedBytes - MaxTensor;
  FResidentLayers := LayerCount;
  if Total > Available then
  begin
    SlotsTotal := 0;
    for I := 0 to High(Sizes) do Inc(SlotsTotal, Sizes[I]);
    if SlotsTotal > Available then
      raise Exception.Create('Insufficient GPU budget for one layer; reduce context length');
    Used := SlotsTotal;
    FResidentLayers := 0;
    while (FResidentLayers < LayerCount) and
      (Layers[FResidentLayers] <= Available - Used) do
    begin
      Inc(Used, Layers[FResidentLayers]); Inc(FResidentLayers);
    end;
    SetLength(FSlots, Length(Sizes));
    for I := 0 to High(Sizes) do
      FSlots[I] := FCompute.CreateGpuBuffer(Sizes[I],
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    FCompute.SynchronousDispatch := True;
  end;
  FStaging := FCompute.CreateGpuBuffer(MaxTensor, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  for I := 0 to FWeights.Count - 1 do
  begin
    W := FWeights[I];
    if W.Layer >= FResidentLayers then Continue;
    W.Buffer := FCompute.CreateGpuBuffer(W.Bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    FWeights[I] := W; // Record ownership before a transfer can fail.
    Upload(W, W.Buffer);
  end;
  if FResidentLayers = LayerCount then FCompute.DestroyGpuBuffer(FStaging);
end;

function TVdxWeightPager.Buffer(const Name: string): TVdxGpuBuffer;
var W: TVdxPagedWeight;
begin
  if not FAllocated then raise Exception.Create('Weight pager is not allocated');
  for W in FWeights do
    if W.Name = Name then
    begin
      if W.Layer < FResidentLayers then Result := W.Buffer
      else Result := FSlots[W.Slot];
      Result.Size := W.Bytes;
      Result.AllocationBytes := 0; // Borrowed handle, never individually freed.
      Exit;
    end;
  raise Exception.Create('Unknown paged weight: ' + Name);
end;

procedure TVdxWeightPager.PrepareLayer(const Layer: Integer);
var W: TVdxPagedWeight;
begin
  if (Layer < FResidentLayers) or (Layer = FCurrentLayer) then Exit;
  FCurrentLayer := -1;
  for W in FWeights do
    if W.Layer = Layer then Upload(W, FSlots[W.Slot]);
  FCurrentLayer := Layer;
end;

function TVdxWeightPager.WeightBytes(): UInt64;
var W: TVdxPagedWeight; B: TVdxGpuBuffer;
begin
  Result := 0;
  for W in FWeights do Inc(Result, W.Buffer.AllocationBytes);
  for B in FSlots do Inc(Result, B.AllocationBytes);
end;

end.
