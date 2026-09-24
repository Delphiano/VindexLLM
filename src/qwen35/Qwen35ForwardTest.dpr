program Qwen35ForwardTest;
{$APPTYPE CONSOLE}
uses System.SysUtils, System.Classes, VindexLLM.Model, VindexLLM.Model.Qwen35,
 VindexLLM.Compute, VindexLLM.Vulkan;
var M: TVdxModel; F: TFileStream; B: TVdxGpuBuffer; Data: TArray<Single>;
 I,J: Integer; Ids: TArray<Integer>;
begin
 try
  M := TVdxModel.LoadModel(ParamStr(1),128);
  if M=nil then raise Exception.Create('Load failed');
  try
   Ids := [9419,1814,0]; SetLength(Data,M.HiddenDim);
   B := M.Compute.CreateGpuBuffer(UInt64(M.HiddenDim)*4,
     VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   F := TFileStream.Create(ParamStr(2),fmCreate);
   try
    for I:=0 to High(Ids) do
    begin
     M.EmbedToken(Ids[I]);
     for J:=0 to Integer(M.NumLayers)-1 do
     begin
      M.Compute.BeginBatch;
      try M.RunLayerForward(J,I); finally M.Compute.EndBatch; end;
      M.Compute.CopyBuffer(M.ResidualMatBuffer,B,UInt64(M.HiddenDim)*4);
      M.Compute.DownloadFromBuffer(B,@Data[0],UInt64(M.HiddenDim)*4);
      F.WriteBuffer(Data[0],UInt64(M.HiddenDim)*4);
     end;
    end;
   finally F.Free; M.Compute.DestroyGpuBuffer(B); end;
  finally M.Free; end;
 except on E:Exception do begin WriteLn(E.Message);Halt(1);end;end;
end.
