program Qwen35StateTest;
{$APPTYPE CONSOLE}
uses System.SysUtils, VindexLLM.Inference, VindexLLM.Sampler;
var E: TVdxInference; C: TVdxSamplerConfig; First, Continued, Restored, Reset: string;
procedure Check(B: Boolean; const S: string);
begin if not B then raise Exception.Create(S+': '+E.GetErrors.ToString); end;
begin
 try
  E := TVdxInference.Create;
  try
   Check(E.LoadModel(ParamStr(1),128),'load');
   C := TVdxSampler.DefaultConfig; C.Temperature := 0; E.SetSamplerConfig(C);
   First := E.Generate('Say hello.',3);
   Check(not E.GetErrors.HasErrors,'generate');
   Check(First<>'','nonempty generation');
   Check(E.SaveKVCache(ParamStr(2)),'save');
   Continued := E.Generate(' Continue.',3,False);
   Check(not E.GetErrors.HasErrors,'continue');
   Check(E.LoadKVCache(ParamStr(2)),'restore');
   Restored := E.Generate(' Continue.',3,False);
   Check(not E.GetErrors.HasErrors,'continue restored');
   Check(Continued=Restored,'snapshot deterministic');
   E.ResetKVCache;
   Reset := E.Generate('Say hello.',3);
   Check(not E.GetErrors.HasErrors,'reset generate');
   Check(First=Reset,'reset deterministic');
   WriteLn('PASS: generation, snapshot/restore, reset');
   WriteLn('ANSWER: ',First);
  finally E.Free; end;
 except on X:Exception do begin WriteLn(X.Message);Halt(1);end;end;
end.
