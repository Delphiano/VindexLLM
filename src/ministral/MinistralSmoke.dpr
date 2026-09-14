program MinistralSmoke;
{$APPTYPE CONSOLE}
uses
  System.SysUtils, System.Classes, System.Diagnostics,
  VindexLLM.Model.Ministral3, VindexLLM.Inference, VindexLLM.Model,
  VindexLLM.GGUFReader, VindexLLM.Tokenizer, VindexLLM.Sampler;
var Engine: TVdxInference; Config: TVdxSamplerConfig; Watch: TStopwatch;
    Reader: TVdxGGUFReader; Tok: TVdxTokenizer; Ids: TArray<Integer>; I: Integer;
    Answer, Prompt: string; Output: TFileStream; Predict: Integer;
begin
  try
    if ParamCount = 0 then raise Exception.Create('Usage: MinistralSmoke model.gguf [prompt] [--tokenize]');
    Prompt := 'Quanto e 2 + 2? Responda apenas o numero.';
    if ParamCount >= 2 then Prompt := ParamStr(2);
    if (ParamCount >= 3) and (ParamStr(3) = '--tokenize') then
    begin
      Reader := TVdxGGUFReader.Create(); Tok := TVdxTokenizer.Create();
      try
        if not Reader.Open(ParamStr(1)) then raise Exception.Create('Cannot open model');
        if not Tok.LoadFromGGUF(Reader) then raise Exception.Create('Cannot load tokenizer');
        Ids := Tok.Encode(Prompt,True);
        for I in Ids do Write(I,','); WriteLn;
        WriteLn(Tok.Decode(Ids));
      finally Tok.Free(); Reader.Free(); end;
      Exit;
    end;
    Engine := TVdxInference.Create();
    try
      Engine.SetStatusCallback(procedure(const S: string; const U: Pointer)
        begin if (Pos('  [',S)<>1) and (Pos('offset=',S)=0) then WriteLn(S); end);
      Watch := TStopwatch.StartNew();
      if not Engine.LoadModel(ParamStr(1),1024) then raise Exception.Create('Load failed');
      Config := TVdxSampler.DefaultConfig(); Config.Temperature := 0;
      Engine.SetSamplerConfig(Config);
      Predict := 32;
      if ParamCount >= 3 then Predict := StrToIntDef(ParamStr(3),32);
      Answer := Engine.Generate(Prompt,Predict);
      if ParamCount >= 4 then
      begin
        Output := TFileStream.Create(ParamStr(4),fmCreate);
        try Output.WriteBuffer(Engine.Model.LogitsVBuf.Memory^,Engine.Model.VocabSize*SizeOf(Single));
        finally Output.Free(); end;
      end;
      WriteLn('ANSWER: ',Answer);
      WriteLn('ELAPSED_MS: ',Watch.ElapsedMilliseconds);
    finally Engine.Free(); end;
  except on E: Exception do begin WriteLn(E.ClassName,': ',E.Message); Halt(1); end; end;
end.
