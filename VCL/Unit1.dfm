object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Form1'
  ClientHeight = 777
  ClientWidth = 1118
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  TextHeight = 15
  object Label1: TLabel
    Left = 23
    Top = 59
    Width = 66
    Height = 15
    Caption = 'Temperatura'
  end
  object Label2: TLabel
    Left = 239
    Top = 59
    Width = 31
    Height = 15
    Caption = 'Top_K'
  end
  object Label3: TLabel
    Left = 23
    Top = 99
    Width = 31
    Height = 15
    Caption = 'Top_P'
  end
  object Label4: TLabel
    Left = 215
    Top = 99
    Width = 33
    Height = 15
    Caption = 'Min_P'
  end
  object Label5: TLabel
    Left = 31
    Top = 139
    Width = 80
    Height = 15
    Caption = 'Repeat_Penalty'
  end
  object Label6: TLabel
    Left = 263
    Top = 139
    Width = 63
    Height = 15
    Caption = 'Max_tokens'
  end
  object Button1: TButton
    Left = 479
    Top = 16
    Width = 75
    Height = 25
    Caption = 'Carregar'
    TabOrder = 0
    OnClick = Button1Click
  end
  object Memo1: TMemo
    Left = 560
    Top = 17
    Width = 531
    Height = 696
    TabOrder = 1
  end
  object Button2: TButton
    Left = 1016
    Top = 728
    Width = 75
    Height = 25
    Caption = 'Descarregar'
    TabOrder = 2
    OnClick = Button2Click
  end
  object Button3: TButton
    Left = 479
    Top = 216
    Width = 75
    Height = 249
    Caption = 'Gerar'
    TabOrder = 3
    OnClick = Button3Click
  end
  object mPrompt: TMemo
    Left = 23
    Top = 216
    Width = 450
    Height = 249
    Lines.Strings = (
      'Estava uma maravilha')
    TabOrder = 4
  end
  object Edit1: TEdit
    Left = 23
    Top = 19
    Width = 450
    Height = 23
    TabOrder = 5
    Text = 'D:\Qwen3.5-0.8B.gguf'
  end
  object edTemperatura: TEdit
    Left = 96
    Top = 51
    Width = 121
    Height = 23
    TabOrder = 6
    Text = '0.1'
  end
  object edTopK: TEdit
    Left = 276
    Top = 51
    Width = 121
    Height = 23
    TabOrder = 7
    Text = '20'
  end
  object edTopP: TEdit
    Left = 60
    Top = 91
    Width = 121
    Height = 23
    TabOrder = 8
    Text = '0.8'
  end
  object edMinP: TEdit
    Left = 252
    Top = 91
    Width = 121
    Height = 23
    TabOrder = 9
    Text = '0.1'
  end
  object edRepeatPenalty: TEdit
    Left = 127
    Top = 136
    Width = 121
    Height = 23
    TabOrder = 10
    Text = '1'
  end
  object Memo2: TMemo
    Left = 23
    Top = 480
    Width = 531
    Height = 273
    TabOrder = 11
  end
  object edMaxTokens: TEdit
    Left = 332
    Top = 136
    Width = 121
    Height = 23
    TabOrder = 12
    Text = '1000'
  end
end
