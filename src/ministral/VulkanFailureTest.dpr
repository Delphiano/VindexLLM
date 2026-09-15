program VulkanFailureTest;
{$APPTYPE CONSOLE}
uses
  System.SysUtils, VindexLLM.Compute, VindexLLM.Vulkan,
  VindexLLM.Model, VindexLLM.Model.Registry, VindexLLM.GGUFReader,
  VindexLLM.Inference;

type
  TFailModel = class(TVdxModel)
    class function SupportedArchitectures(): TArray<string>; override;
    function InitSubsystems(): Boolean; override;
    function LoadWeights(): Boolean; override;
  end;

class function TFailModel.SupportedArchitectures(): TArray<string>;
begin Result := ['mistral3']; end;

function TFailModel.InitSubsystems(): Boolean;
begin FCompute.Init(); Result := FCompute.IsReady(); end;

function TFailModel.LoadWeights(): Boolean;
var B: TVdxGpuBuffer;
begin
  // No Vulkan memory type can satisfy this reserved property bit.
  B := FCompute.CreateGpuBuffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, $80000000);
  FCompute.DestroyGpuBuffer(B);
  Result := True;
end;

procedure Check(const OK: Boolean; const Msg: string);
begin if not OK then raise Exception.Create(Msg); end;

var C: TVdxCompute; A, B: TVdxGpuBuffer; X, Y: UInt32;
    RaisedError: Boolean; Engine: TVdxInference; ErrorText: string;
    Reader: TVdxGGUFReader; Info: TVdxGGUFTensorInfo; Total, Rows: UInt64; I: Integer;
begin
  try
    C := TVdxCompute.Create();
    try
      A := Default(TVdxGpuBuffer);
      RaisedError := False;
      try C.UploadToBuffer(A, nil, 4);
      except on E: EVdxVulkanError do RaisedError := True; end;
      Check(RaisedError, 'Uninitialized upload must raise a Vulkan error');
    finally C.Free(); end;

    C := TVdxCompute.Create();
    A := Default(TVdxGpuBuffer); B := Default(TVdxGpuBuffer);
    try
      C.Init(); Check(C.IsReady(), 'GPU initialization failed');
      WriteLn('GPU: ', C.GetDeviceName());
      A := C.CreateGpuBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      B := C.CreateGpuBuffer(4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      X := $1234ABCD; Y := 0;
      C.UploadToBuffer(A, @X, 4); C.CopyBuffer(A, B, 4); C.DownloadFromBuffer(B, @Y, 4);
      Check(X = Y, 'GPU round trip mismatch');
      RaisedError := False;
      try C.CopyBuffer(A, B, 8);
      except on E: EVdxVulkanError do RaisedError := True; end;
      Check(RaisedError, 'Out-of-range copy must fail before GPU submission');
      C.GetErrors().Clear();
      Check(not C.IsReady(), 'Clearing errors must not reset faulted GPU state');
      RaisedError := False;
      try C.BeginBatch();
      except on E: EVdxVulkanError do RaisedError := True; end;
      Check(RaisedError, 'Faulted GPU must reject subsequent generation commands');
    finally C.DestroyGpuBuffer(B); C.DestroyGpuBuffer(A); C.Free(); end;

    Check(ParamCount >= 1, 'Pass Ministral GGUF for model failure test');
    TVdxModelRegistry.RegisterClass(TFailModel);
    Engine := TVdxInference.Create();
    try
      Check(not Engine.LoadModel(ParamStr(1), 32), 'Injected memory failure must reject model');
      Check(Engine.Model = nil, 'Partial model must be freed');
      ErrorText := Engine.GetErrors().ToString();
      Check(Pos('No suitable memory type', ErrorText) > 0, 'Original allocation error was lost');
      Check(Engine.Generate('hello', 1) = '', 'Generation must reject failed model');
      Check(Engine.GetErrors().ToString() = ErrorText, 'Generation lost original load diagnostic');
      // A second attempt also exercises cleanup after a partial load.
      Check(not Engine.LoadModel(ParamStr(1), 32), 'Reload must also report injected failure');
      WriteLn('EXPECTED ERROR: ', Engine.GetErrors().ToString());
    finally Engine.Free(); end;

    if ParamCount >= 2 then
    begin
      Reader := TVdxGGUFReader.Create();
      try
        Check(Reader.Open(ParamStr(2)), 'Cannot inspect Gemma GGUF'); Total := 0;
        for Info in Reader.GetTensorList() do
        begin
          Rows := 1;
          for I := 1 to Integer(Info.NumDimensions) - 1 do Rows := Rows * Info.Dimensions[I];
          Inc(Total, VdxGGMLTensorBytes(Info.TensorType, Info.Dimensions[0], Rows));
        end;
        WriteLn('GGUF tensor bytes: ', Total, ' (', Total / 1073741824:0:3, ' GiB)');
      finally Reader.Free(); end;
    end;
    WriteLn('PASS: GPU copy, invalid ranges, uninitialized/faulted device, partial load cleanup and preserved errors');
  except on E: Exception do begin WriteLn(E.ClassName, ': ', E.Message); Halt(1); end; end;
end.
