{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Tokenizer;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  VindexLLM.Utils,
  VindexLLM.GGUFReader;

type

  { TVdxTokenType }
  TVdxTokenType = (
    ttNormal = 1,
    ttUnknown = 2,
    ttControl = 3,
    ttUserDefined = 4,
    ttUnused = 5,
    ttByte = 6
  );

  { TVdxTokenizer }
  TVdxTokenizer = class(TVdxBaseObject)
  private
    FTokens: TArray<string>;         // Token strings indexed by ID
    FScores: TArray<Single>;         // BPE merge scores indexed by ID
    FTypes: TArray<Integer>;         // Token type flags indexed by ID
    FVocabSize: Integer;
    FBosId: Integer;
    FEosId: Integer;
    FByteBPE: Boolean;
    FByteChars: array[0..255] of Char;
    FCharBytes: TDictionary<Char, Byte>;
    FMergeRanks: TDictionary<string, Integer>;
    FPendingUTF8: TBytes;

    // Lookup: token string -> ID (for encoding)
    FTokenToId: TDictionary<string, Integer>;

    // Special/control tokens sorted by length descending (for greedy match)
    FSpecialTokens: TArray<TPair<string, Integer>>;

    procedure EncodeByteSegment(const AText: string; const AResult: TList<Integer>);
    function TokenBytes(const AId: Integer): TBytes;

    // BPE merge: find best pair to merge in a token list
    function FindBestMerge(const APieces: TList<Integer>): Integer;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Load vocabulary from GGUF reader
    function LoadFromGGUF(const AReader: TVdxGGUFReader): Boolean;

    // Encode text to token IDs (adds BOS automatically)
    function Encode(const AText: string; const AAddBos: Boolean = True): TArray<Integer>;

    // Decode token IDs back to text
    function Decode(const AIds: TArray<Integer>): string;
    procedure ResetDecoder();
    function DecodeToken(const AId: Integer): string;

    // Accessors
    function GetVocabSize(): Integer;
    function GetBosId(): Integer;
    function GetEosId(): Integer;
    function GetTokenStr(const AId: Integer): string;
  end;

implementation

uses
  System.Math, System.RegularExpressions;

{ TVdxTokenizer }
constructor TVdxTokenizer.Create();
begin
  inherited;
  FTokenToId := TDictionary<string, Integer>.Create();
  FCharBytes := TDictionary<Char, Byte>.Create();
  FMergeRanks := TDictionary<string, Integer>.Create();
  FVocabSize := 0;
  FBosId := 2;
  FEosId := 1;
end;

destructor TVdxTokenizer.Destroy();
begin
  FCharBytes.Free();
  FMergeRanks.Free();
  FTokenToId.Free();
  inherited;
end;

function TVdxTokenizer.LoadFromGGUF(const AReader: TVdxGGUFReader): Boolean;
var
  LVocab: TVdxGGUFMetaValue;
  LScoresVal: TVdxGGUFMetaValue;
  LTypesVal: TVdxGGUFMetaValue;
  LI: Integer;
  LSpecialList: TList<TPair<string, Integer>>;
  LTokenType: Integer;
  LNext: Integer;
  LMerges: TVdxGGUFMetaValue;
begin
  Result := False;

  if AReader = nil then
  begin
    FErrors.Add(esFatal, 'TOKENIZER', 'LoadFromGGUF called with nil reader');
    Exit;
  end;

  if not AReader.HasMetadata('tokenizer.ggml.tokens') then
  begin
    FErrors.Add(esFatal, 'TOKENIZER',
      'GGUF missing required key: tokenizer.ggml.tokens');
    Exit;
  end;

  FByteBPE := SameText(AReader.GetMetadataString('tokenizer.ggml.model'), 'gpt2');
  FMergeRanks.Clear();
  if FByteBPE then
  begin
    if not SameText(AReader.GetMetadataString('tokenizer.ggml.pre'), 'tekken') then
      raise ENotSupportedException.Create('Only Tekken byte-BPE is implemented');
    if not AReader.GetMetadata('tokenizer.ggml.merges', LMerges) then
      raise EConvertError.Create('Tekken vocabulary requires merge ranks');
    for LI := 0 to High(LMerges.ArrayItems) do
      if not FMergeRanks.ContainsKey(LMerges.ArrayItems[LI].AsString) then
        FMergeRanks.Add(LMerges.ArrayItems[LI].AsString, LI);
    // GPT-2 reversible byte alphabet used by GGUF's Tekken conversion.
    FCharBytes.Clear();
    LNext := 256;
    for LI := 0 to 255 do
    begin
      if ((LI >= 33) and (LI <= 126)) or ((LI >= 161) and (LI <= 172)) or (LI >= 174) then
        FByteChars[LI] := Char(LI)
      else
      begin
        FByteChars[LI] := Char(LNext);
        Inc(LNext);
      end;
      FCharBytes.Add(FByteChars[LI], Byte(LI));
    end;
  end;
  ResetDecoder();

  // Read token strings
  if not AReader.GetMetadata('tokenizer.ggml.tokens', LVocab) then
    Exit;
  FVocabSize := Length(LVocab.ArrayItems);
  SetLength(FTokens, FVocabSize);
  for LI := 0 to FVocabSize - 1 do
    FTokens[LI] := LVocab.ArrayItems[LI].AsString;

  // Read scores
  SetLength(FScores, FVocabSize);
  if AReader.HasMetadata('tokenizer.ggml.scores') then
  begin
    AReader.GetMetadata('tokenizer.ggml.scores', LScoresVal);
    for LI := 0 to FVocabSize - 1 do
      FScores[LI] := LScoresVal.ArrayItems[LI].AsFloat64;
  end;

  // Read token types
  SetLength(FTypes, FVocabSize);
  if AReader.HasMetadata('tokenizer.ggml.token_type') then
  begin
    AReader.GetMetadata('tokenizer.ggml.token_type', LTypesVal);
    for LI := 0 to FVocabSize - 1 do
      FTypes[LI] := Integer(LTypesVal.ArrayItems[LI].AsInt64);
  end;

  // Build token-to-id lookup
  FTokenToId.Clear();
  for LI := 0 to FVocabSize - 1 do
    FTokenToId.AddOrSetValue(FTokens[LI], LI);

  // Build special token list (control + user-defined), sorted by length desc
  // so longer matches are tried first
  LSpecialList := TList<TPair<string, Integer>>.Create();
  try
    for LI := 0 to FVocabSize - 1 do
    begin
      LTokenType := FTypes[LI];
      if (LTokenType = Ord(ttControl)) or (LTokenType = Ord(ttUserDefined)) then
      begin
        if FTokens[LI] <> '' then
          LSpecialList.Add(TPair<string, Integer>.Create(FTokens[LI], LI));
      end;
    end;

    // Sort by string length descending (greedy match longest first)
    LSpecialList.Sort(TComparer<TPair<string, Integer>>.Construct(
      function(const ALeft: TPair<string, Integer>;
        const ARight: TPair<string, Integer>): Integer
      begin
        Result := Length(ARight.Key) - Length(ALeft.Key);
      end
    ));

    FSpecialTokens := LSpecialList.ToArray();
  finally
    LSpecialList.Free();
  end;

  // Read BOS/EOS IDs from metadata
  FBosId := Integer(AReader.GetMetadataUInt32('tokenizer.ggml.bos_token_id', 2));
  FEosId := Integer(AReader.GetMetadataUInt32('tokenizer.ggml.eos_token_id', 1));

  Result := True;
end;

function TVdxTokenizer.FindBestMerge(const APieces: TList<Integer>): Integer;
var
  LI: Integer;
  LMergedStr: string;
  LMergedId: Integer;
  LBestIdx: Integer;
  LBestScore: Single;
  LScore: Single;
begin
  LBestIdx := -1;
  LBestScore := -1e30;

  for LI := 0 to APieces.Count - 2 do
  begin
    LMergedStr := FTokens[APieces[LI]] + FTokens[APieces[LI + 1]];
    if FTokenToId.TryGetValue(LMergedStr, LMergedId) then
    begin
      LScore := FScores[LMergedId];
      if LScore > LBestScore then
      begin
        LBestScore := LScore;
        LBestIdx := LI;
      end;
    end;
  end;

  Result := LBestIdx;
end;

function TVdxTokenizer.Encode(const AText: string;
  const AAddBos: Boolean): TArray<Integer>;
var
  LResult: TList<Integer>;
  LPos: Integer;
  LTextLen: Integer;
  LMatched: Boolean;
  LI: Integer;
  LSpecStr: string;
  LSpecLen: Integer;
  LSegEnd: Integer;
  LSegment: string;
  LNormalized: string;
  LCharStr: string;
  LCharId: Integer;
  LPieces: TList<Integer>;
  LMergeIdx: Integer;
  LMergedStr: string;
  LMergedId: Integer;
  LCharIdx: Integer;
  LBytes: TBytes;
  LByte: Byte;
  LByteToken: string;
begin
  LResult := TList<Integer>.Create();
  try
    if AAddBos then
      LResult.Add(FBosId);

    LPos := 1;  // Delphi strings are 1-based
    LTextLen := Length(AText);

    while LPos <= LTextLen do
    begin
      // Try to match a special token at current position
      LMatched := False;
      for LI := 0 to Length(FSpecialTokens) - 1 do
      begin
        LSpecStr := FSpecialTokens[LI].Key;
        LSpecLen := Length(LSpecStr);
        if (LPos + LSpecLen - 1 <= LTextLen) and
           (Copy(AText, LPos, LSpecLen) = LSpecStr) then
        begin
          LResult.Add(FSpecialTokens[LI].Value);
          LPos := LPos + LSpecLen;
          LMatched := True;
          Break;
        end;
      end;

      if LMatched then
        Continue;

      // Find extent of regular text (until next special token or end)
      LSegEnd := LPos + 1;
      while LSegEnd <= LTextLen do
      begin
        LMatched := False;
        for LI := 0 to Length(FSpecialTokens) - 1 do
        begin
          LSpecStr := FSpecialTokens[LI].Key;
          LSpecLen := Length(LSpecStr);
          if (LSegEnd + LSpecLen - 1 <= LTextLen) and
             (Copy(AText, LSegEnd, LSpecLen) = LSpecStr) then
          begin
            LMatched := True;
            Break;
          end;
        end;
        if LMatched then
          Break;
        Inc(LSegEnd);
      end;

      // Extract regular text segment
      LSegment := Copy(AText, LPos, LSegEnd - LPos);
      LPos := LSegEnd;

      if FByteBPE then
      begin
        EncodeByteSegment(LSegment, LResult);
        Continue;
      end;

      // Normalize: replace spaces with ▁ (U+2581)
      LNormalized := LSegment.Replace(' ', #$2581);

      // Split into individual characters, map each to a token ID
      LPieces := TList<Integer>.Create();
      try
        LCharIdx := 1;
        while LCharIdx <= Length(LNormalized) do
        begin
          // Get one character (handle surrogate pairs for chars > U+FFFF)
          if (LCharIdx < Length(LNormalized)) and
             (Ord(LNormalized[LCharIdx]) >= $D800) and
             (Ord(LNormalized[LCharIdx]) <= $DBFF) and
             (Ord(LNormalized[LCharIdx + 1]) >= $DC00) and
             (Ord(LNormalized[LCharIdx + 1]) <= $DFFF) then
          begin
            LCharStr := Copy(LNormalized, LCharIdx, 2);
            Inc(LCharIdx, 2);
          end
          else
          begin
            LCharStr := LNormalized[LCharIdx];
            Inc(LCharIdx);
          end;

          // Look up single character in vocab
          if FTokenToId.TryGetValue(LCharStr, LCharId) then
          begin
            LPieces.Add(LCharId);
          end
          else
          begin
            // Byte fallback: encode character as UTF-8 bytes
            LBytes := TEncoding.UTF8.GetBytes(LCharStr);
            for LByte in LBytes do
            begin
              // Byte tokens in GGUF are stored as <0xHH>
              LByteToken := Format('<0x%s>', [IntToHex(LByte, 2)]);
              if FTokenToId.TryGetValue(LByteToken, LCharId) then
                LPieces.Add(LCharId);
            end;
          end;
        end;

        // BPE merge loop: repeatedly merge the best scoring pair
        LMergeIdx := FindBestMerge(LPieces);
        while LMergeIdx >= 0 do
        begin
          LMergedStr := FTokens[LPieces[LMergeIdx]] +
                        FTokens[LPieces[LMergeIdx + 1]];
          if FTokenToId.TryGetValue(LMergedStr, LMergedId) then
          begin
            LPieces[LMergeIdx] := LMergedId;
            LPieces.Delete(LMergeIdx + 1);
          end
          else
            Break;  // Should not happen, but safety exit
          LMergeIdx := FindBestMerge(LPieces);
        end;

        // Add merged pieces to result
        for LI := 0 to LPieces.Count - 1 do
          LResult.Add(LPieces[LI]);
      finally
        LPieces.Free();
      end;
    end;

    Result := LResult.ToArray();
  finally
    LResult.Free();
  end;
end;

function TVdxTokenizer.Decode(const AIds: TArray<Integer>): string;
var
  LI: Integer;
  LId: Integer;
  LToken: string;
  LBytes: TBytes;
begin
  if FByteBPE then
  begin
    LBytes := nil;
    for LId in AIds do LBytes := LBytes + TokenBytes(LId);
    Exit(TEncoding.UTF8.GetString(LBytes));
  end;
  Result := '';
  for LI := 0 to Length(AIds) - 1 do
  begin
    LId := AIds[LI];
    if (LId >= 0) and (LId < FVocabSize) then
    begin
      LToken := FTokens[LId];
      // Replace ▁ back to space for display
      LToken := LToken.Replace(#$2581, ' ');
      Result := Result + LToken;
    end;
  end;
end;

procedure TVdxTokenizer.EncodeByteSegment(const AText: string;
  const AResult: TList<Integer>);
const
  // Exact Unicode-aware Tekken pre-tokenizer (mistral-common/tokenizer.json).
  Pattern = '[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+' +
    '|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*' +
    '|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+';
var
  M: TMatch;
  Pieces: TList<string>;
  Bytes: TBytes;
  B: Byte;
  I, Rank, Best, BestRank, Id, Consumed: Integer;
  Key: string;
begin
  Consumed := 0;
  for M in TRegEx.Matches(AText, Pattern) do
  begin
    if M.Index <> Consumed + 1 then raise EConvertError.Create('Tekken pre-tokenizer skipped input');
    Inc(Consumed, M.Length);
    Bytes := TEncoding.UTF8.GetBytes(M.Value);
    // Tekken's ignore_merges flag: an entire pre-token present in the vocab wins.
    SetLength(Key, Length(Bytes));
    for I := 0 to High(Bytes) do Key[I+1] := FByteChars[Bytes[I]];
    if FTokenToId.TryGetValue(Key, Id) then
    begin
      AResult.Add(Id);
      Continue;
    end;
    Pieces := TList<string>.Create();
    try
      for B in Bytes do Pieces.Add(string(FByteChars[B]));
      repeat
        Best := -1;
        BestRank := MaxInt;
        for I := 0 to Pieces.Count - 2 do
        begin
          Key := Pieces[I] + ' ' + Pieces[I + 1];
          if FMergeRanks.TryGetValue(Key, Rank) and (Rank < BestRank) then
          begin
            Best := I;
            BestRank := Rank;
          end;
        end;
        if Best < 0 then Break;
        Pieces[Best] := Pieces[Best] + Pieces[Best + 1];
        Pieces.Delete(Best + 1);
      until False;
      for I := 0 to Pieces.Count - 1 do
      begin
        if not FTokenToId.TryGetValue(Pieces[I], Id) then
          raise EConvertError.Create('Tekken piece missing from GGUF vocabulary');
        AResult.Add(Id);
      end;
    finally
      Pieces.Free();
    end;
  end;
  if Consumed <> Length(AText) then raise EConvertError.Create('Tekken pre-tokenizer left unparsed input');
end;

function TVdxTokenizer.TokenBytes(const AId: Integer): TBytes;
var I: Integer; S: string; B: Byte;
begin
  if (AId < 0) or (AId >= FVocabSize) then raise ERangeError.Create('Invalid token ID');
  S := FTokens[AId];
  if not FByteBPE or (FTypes[AId] = Ord(ttControl)) or (FTypes[AId] = Ord(ttUserDefined)) then
    Exit(TEncoding.UTF8.GetBytes(S));
  SetLength(Result, Length(S));
  for I := 1 to Length(S) do
  begin
    if not FCharBytes.TryGetValue(S[I], B) then raise EConvertError.Create('Invalid byte-BPE alphabet');
    Result[I-1] := B;
  end;
end;

procedure TVdxTokenizer.ResetDecoder();
begin
  FPendingUTF8 := nil;
end;

function TVdxTokenizer.DecodeToken(const AId: Integer): string;
var I, N, J: Integer;
begin
  if not FByteBPE then Exit(Decode(TArray<Integer>.Create(AId)));
  FPendingUTF8 := FPendingUTF8 + TokenBytes(AId);
  I := 0;
  while I < Length(FPendingUTF8) do
  begin
    if FPendingUTF8[I] < $80 then N := 1
    else if FPendingUTF8[I] < $E0 then N := 2
    else if FPendingUTF8[I] < $F0 then N := 3
    else N := 4;
    if I + N > Length(FPendingUTF8) then Break;
    for J := 1 to N-1 do
      if (FPendingUTF8[I+J] and $C0) <> $80 then
      begin N := 1; Break; end;
    Inc(I, N);
  end;
  Result := TEncoding.UTF8.GetString(FPendingUTF8, 0, I);
  FPendingUTF8 := Copy(FPendingUTF8, I, Length(FPendingUTF8)-I);
end;

function TVdxTokenizer.GetVocabSize(): Integer;
begin
  Result := FVocabSize;
end;

function TVdxTokenizer.GetBosId(): Integer;
begin
  Result := FBosId;
end;

function TVdxTokenizer.GetEosId(): Integer;
begin
  Result := FEosId;
end;

function TVdxTokenizer.GetTokenStr(const AId: Integer): string;
begin
  if (AId >= 0) and (AId < FVocabSize) then
    Result := FTokens[AId]
  else
    Result := '<invalid>';
end;

end.
