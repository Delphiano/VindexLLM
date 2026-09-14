program MinistralKernelTest;
{$APPTYPE CONSOLE}
uses System.SysUtils, System.Classes, System.IOUtils, Winapi.Windows,
 VindexLLM.Compute,VindexLLM.Vulkan,VindexLLM.Model.Ministral3;
var GPU: TVdxCompute; Layout: VkDescriptorSetLayout; Module: VkShaderModule;
 Bundle: TVdxComputePipelineBundle; Pool: VkDescriptorPool; Desc: VkDescriptorSet;
 S: TResourceStream; F: TFileStream; P: TVdxMinistralPush; Groups: array[0..2] of UInt32;
 Sizes: array[0..3] of UInt32; Buffers: array[0..3] of TVdxGpuBuffer;
 Data: TBytes; I: Integer; Name: string;
begin
 try
  GPU := TVdxCompute.Create();
  try
   GPU.Init();
   if GPU.GetErrors().HasErrors() then raise Exception.Create(GPU.GetErrors().ToString());
   S := TResourceStream.Create(HInstance,'MINISTRAL_DENSE',RT_RCDATA);
   try Module := GPU.CreateShaderModule(S.Memory,S.Size); finally S.Free(); end;
   Layout := GPU.CreateStorageDescriptorSetLayout(4);
   Bundle := GPU.CreateComputePipelineWithPush(Module,'main',Layout,SizeOf(P));
   for Name in TDirectory.GetFiles(ParamStr(1),'*.case') do
   begin
    F := TFileStream.Create(Name,fmOpenRead);
    try
     F.ReadBuffer(Groups,SizeOf(Groups));F.ReadBuffer(Sizes,SizeOf(Sizes));F.ReadBuffer(P,SizeOf(P));
     for I := 0 to 3 do
     begin
      Buffers[I] := GPU.CreateGpuBuffer(Sizes[I],VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      SetLength(Data,Sizes[I]);F.ReadBuffer(Data[0],Sizes[I]);GPU.UploadToBuffer(Buffers[I],@Data[0],Sizes[I]);
     end;
    finally F.Free(); end;
    Pool := GPU.CreateDescriptorPoolForStorage(1,4);
    Desc := GPU.AllocateDescriptorSetForBuffers(Pool,Layout,Buffers);
    GPU.DispatchComputeWithPush(Bundle.Pipeline,Bundle.PipelineLayout,Desc,@P,SizeOf(P),Groups[0],Groups[1],Groups[2]);
    if GPU.GetErrors().HasErrors() then raise Exception.Create(GPU.GetErrors().ToString());
    SetLength(Data,Sizes[3]);GPU.DownloadFromBuffer(Buffers[3],@Data[0],Sizes[3]);
    TFile.WriteAllBytes(ChangeFileExt(Name,'.actual'),Data);
    GPU.DestroyDescriptorPoolHandle(Pool);
    for I := 0 to 3 do GPU.DestroyGpuBuffer(Buffers[I]);
    WriteLn(ExtractFileName(Name),' OK');
   end;
   GPU.DestroyComputePipelineBundle(Bundle);GPU.DestroyDescriptorSetLayoutHandle(Layout);GPU.DestroyShaderModuleHandle(Module);
  finally GPU.Free(); end;
 except on E:Exception do begin WriteLn(E.ClassName,': ',E.Message);Halt(1);end;end;
end.
