' Bloomberg Data Extractor for Smart Wheel Engine
' ================================================
' This VBA module automates Bloomberg data extraction via Excel.
'
' SETUP:
' 1. Open Excel with Bloomberg Add-in loaded
' 2. Press Alt+F11 to open VBA Editor
' 3. Insert > Module
' 4. Paste this entire file
' 5. Close VBA Editor
' 6. Run macros from Developer > Macros (or Alt+F8)
'
' USAGE:
' - ExtractAllData: Extracts all data types for all tickers
' - ExtractOHLCV: Extracts just OHLCV price history
' - ExtractOptions: Extracts option chains
' - ExtractIVHistory: Extracts IV history for IV rank
'
' OUTPUT:
' Files are saved to the paths specified in BLOOMBERG_DIR constant.
' Default: C:\SmartWheelEngine\data\bloomberg\

Option Explicit

' ============ CONFIGURATION ============
' Modify these paths to match your setup
Private Const BLOOMBERG_DIR As String = "C:\SmartWheelEngine\data\bloomberg\"
Private Const START_DATE As String = "20190101"

' Tickers to extract (modify this list as needed)
Private Function GetTickers() As Variant
    GetTickers = Array("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", _
                       "JPM", "BAC", "WFC", "GS", _
                       "UNH", "JNJ", "LLY", "ABBV", _
                       "XOM", "CVX", _
                       "PG", "KO", "HD", "MCD", _
                       "CAT", "HON", "GE")
End Function

' ============ MAIN EXTRACTION ROUTINES ============

Public Sub ExtractAllData()
    ' Extract all data types for all tickers
    Dim startTime As Double
    startTime = Timer

    MsgBox "Starting full data extraction. This may take 30-60 minutes.", vbInformation

    ExtractOHLCV
    ExtractIVHistory
    ExtractEarnings
    ExtractDividends
    ExtractRates
    ExtractFundamentals

    MsgBox "Extraction complete! Time: " & Format(Timer - startTime, "0.0") & " seconds", vbInformation
End Sub

Public Sub ExtractOHLCV()
    ' Extract OHLCV price history for all tickers
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    For Each ticker In tickers
        ' Create new worksheet
        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_OHLCV", 31)

        ' BDH formula for OHLCV
        ws.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""" & START_DATE & """,""" & endDate & """,""Dir=V"")"

        ' Wait for data
        WaitForBloomberg ws.Range("A1"), 30

        ' Add headers
        ws.Range("A1").EntireRow.Insert
        ws.Range("A1:F1").Value = Array("Date", "Open", "High", "Low", "Close", "Volume")

        ' Save to CSV
        outputPath = BLOOMBERG_DIR & "ohlcv\" & ticker & ".csv"
        EnsureDirectoryExists BLOOMBERG_DIR & "ohlcv\"
        SaveAsCSV ws, outputPath

        ' Clean up
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True

        Debug.Print "Extracted OHLCV for " & ticker
    Next ticker

    Application.ScreenUpdating = True
    MsgBox "OHLCV extraction complete!", vbInformation
End Sub

Public Sub ExtractIVHistory()
    ' Extract IV history for IV rank calculation
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String
    Dim formula As String

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    For Each ticker In tickers
        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_IV", 31)

        ' BDH formula for IV metrics
        formula = "=BDH(""" & ticker & " US Equity""," & _
                  """30DAY_IMPVOL_100.0%MNY_DF,60DAY_IMPVOL_100.0%MNY_DF," & _
                  "30DAY_IMPVOL_90.0%MNY_DF,30DAY_IMPVOL_110.0%MNY_DF," & _
                  "20DAY_HV,60DAY_HV""," & _
                  """" & START_DATE & """,""" & endDate & """,""Dir=V"")"

        ws.Range("A1").Formula = formula

        WaitForBloomberg ws.Range("A1"), 30

        ' Add headers
        ws.Range("A1").EntireRow.Insert
        ws.Range("A1:G1").Value = Array("Date", "IV_30D", "IV_60D", "IV_25D_Put", "IV_25D_Call", "HV_20D", "HV_60D")

        outputPath = BLOOMBERG_DIR & "iv_history\" & ticker & ".csv"
        EnsureDirectoryExists BLOOMBERG_DIR & "iv_history\"
        SaveAsCSV ws, outputPath

        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True

        Debug.Print "Extracted IV history for " & ticker
    Next ticker

    Application.ScreenUpdating = True
    MsgBox "IV history extraction complete!", vbInformation
End Sub

Public Sub ExtractEarnings()
    ' Extract earnings history
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    For Each ticker In tickers
        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_ERN", 31)

        ' BDH for quarterly EPS data
        ws.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""IS_EPS,BEST_EPS_MEDIAN,EARN_EST_EPS_SURPRISE_PCT"",""" & START_DATE & """,""" & endDate & """,""Dir=V"",""Per=Q"")"

        WaitForBloomberg ws.Range("A1"), 30

        ws.Range("A1").EntireRow.Insert
        ws.Range("A1:D1").Value = Array("Date", "EPS_Actual", "EPS_Estimate", "Surprise_Pct")

        outputPath = BLOOMBERG_DIR & "earnings\" & ticker & ".csv"
        EnsureDirectoryExists BLOOMBERG_DIR & "earnings\"
        SaveAsCSV ws, outputPath

        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True

        Debug.Print "Extracted earnings for " & ticker
    Next ticker

    Application.ScreenUpdating = True
    MsgBox "Earnings extraction complete!", vbInformation
End Sub

Public Sub ExtractDividends()
    ' Extract dividend history
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    For Each ticker In tickers
        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_DIV", 31)

        ' BDS for dividend history
        ws.Range("A1").Formula = "=BDS(""" & ticker & " US Equity"",""DVD_HIST_ALL"",""DVD_START_DT=" & START_DATE & """,""DVD_END_DT=" & endDate & """)"

        WaitForBloomberg ws.Range("A1"), 30

        outputPath = BLOOMBERG_DIR & "dividends\" & ticker & ".csv"
        EnsureDirectoryExists BLOOMBERG_DIR & "dividends\"
        SaveAsCSV ws, outputPath

        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True

        Debug.Print "Extracted dividends for " & ticker
    Next ticker

    Application.ScreenUpdating = True
    MsgBox "Dividend extraction complete!", vbInformation
End Sub

Public Sub ExtractRates()
    ' Extract Treasury yields
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String

    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "Treasury_Yields"

    ' 3-month T-bill
    ws.Range("A1").Formula = "=BDH(""USGG3M Index"",""PX_LAST"",""" & START_DATE & """,""" & endDate & """,""Dir=V"")"

    WaitForBloomberg ws.Range("A1"), 30

    ws.Range("A1").EntireRow.Insert
    ws.Range("A1:B1").Value = Array("Date", "Rate_3M")

    outputPath = BLOOMBERG_DIR & "rates\treasury_yields.csv"
    EnsureDirectoryExists BLOOMBERG_DIR & "rates\"
    SaveAsCSV ws, outputPath

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    Application.ScreenUpdating = True
    MsgBox "Rates extraction complete!", vbInformation
End Sub

Public Sub ExtractFundamentals()
    ' Extract company fundamentals for all tickers
    Dim tickers As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim i As Long

    tickers = GetTickers()

    Application.ScreenUpdating = False

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "Fundamentals"

    ' Headers
    ws.Range("A1:G1").Value = Array("Ticker", "Market_Cap", "GICS_Sector", "GICS_Industry", "Div_Yield", "PE", "EPS")

    ' Fill ticker column
    For i = LBound(tickers) To UBound(tickers)
        ws.Cells(i + 2, 1).Value = tickers(i) & " US Equity"
    Next i

    ' BDP formulas for each field
    For i = LBound(tickers) To UBound(tickers)
        Dim row As Long
        row = i + 2
        ws.Cells(row, 2).Formula = "=BDP(A" & row & ",""CUR_MKT_CAP"")"
        ws.Cells(row, 3).Formula = "=BDP(A" & row & ",""GICS_SECTOR_NAME"")"
        ws.Cells(row, 4).Formula = "=BDP(A" & row & ",""GICS_INDUSTRY_GROUP_NAME"")"
        ws.Cells(row, 5).Formula = "=BDP(A" & row & ",""EQY_DVD_YLD_IND"")"
        ws.Cells(row, 6).Formula = "=BDP(A" & row & ",""PE_RATIO"")"
        ws.Cells(row, 7).Formula = "=BDP(A" & row & ",""TRAIL_12M_EPS"")"
    Next i

    WaitForBloomberg ws.Cells(2, 2), 60

    outputPath = BLOOMBERG_DIR & "fundamentals\sp500_fundamentals.csv"
    EnsureDirectoryExists BLOOMBERG_DIR & "fundamentals\"
    SaveAsCSV ws, outputPath

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    Application.ScreenUpdating = True
    MsgBox "Fundamentals extraction complete!", vbInformation
End Sub

Public Sub ExtractOptionChain()
    ' Extract option chain for a single ticker (interactive)
    Dim ticker As String
    Dim ws As Worksheet
    Dim outputPath As String
    Dim today As String

    ticker = InputBox("Enter ticker symbol (e.g., AAPL):", "Option Chain Extraction", "AAPL")
    If ticker = "" Then Exit Sub

    today = Format(Date, "YYYY-MM-DD")

    Application.ScreenUpdating = False

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = Left(ticker & "_OPT", 31)

    ' Get option chain tickers
    ws.Range("A1").Formula = "=BDS(""" & ticker & " US Equity"",""OPT_CHAIN"")"

    WaitForBloomberg ws.Range("A1"), 60

    ' Count options
    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).row

    If lastRow < 2 Then
        MsgBox "No options found for " & ticker, vbExclamation
        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True
        Exit Sub
    End If

    ' Add BDP formulas for each option
    ws.Range("B1").Value = "Strike"
    ws.Range("C1").Value = "Type"
    ws.Range("D1").Value = "Expiry"
    ws.Range("E1").Value = "Bid"
    ws.Range("F1").Value = "Ask"
    ws.Range("G1").Value = "IV"
    ws.Range("H1").Value = "Delta"
    ws.Range("I1").Value = "OI"
    ws.Range("J1").Value = "Volume"

    Dim r As Long
    For r = 2 To lastRow
        ws.Cells(r, 2).Formula = "=BDP(A" & r & ",""OPT_STRIKE_PX"")"
        ws.Cells(r, 3).Formula = "=BDP(A" & r & ",""OPT_PUT_CALL"")"
        ws.Cells(r, 4).Formula = "=BDP(A" & r & ",""OPT_EXPIRE_DT"")"
        ws.Cells(r, 5).Formula = "=BDP(A" & r & ",""BID"")"
        ws.Cells(r, 6).Formula = "=BDP(A" & r & ",""ASK"")"
        ws.Cells(r, 7).Formula = "=BDP(A" & r & ",""OPT_IMPLIED_VOLATILITY_MID"")"
        ws.Cells(r, 8).Formula = "=BDP(A" & r & ",""OPT_DELTA"")"
        ws.Cells(r, 9).Formula = "=BDP(A" & r & ",""OPEN_INT"")"
        ws.Cells(r, 10).Formula = "=BDP(A" & r & ",""VOLUME"")"
    Next r

    WaitForBloomberg ws.Cells(2, 2), 120

    outputPath = BLOOMBERG_DIR & "options\" & today & "_" & ticker & ".csv"
    EnsureDirectoryExists BLOOMBERG_DIR & "options\"
    SaveAsCSV ws, outputPath

    Application.ScreenUpdating = True
    MsgBox "Option chain saved to " & outputPath, vbInformation
End Sub

' ============ HELPER FUNCTIONS ============

Private Sub WaitForBloomberg(cell As Range, timeoutSeconds As Long)
    ' Wait for Bloomberg data to populate
    Dim startTime As Double
    startTime = Timer

    Do While Timer - startTime < timeoutSeconds
        DoEvents
        Application.Wait (Now + TimeValue("0:00:01"))

        ' Check if data has arrived
        If Not IsError(cell.Value) Then
            If cell.Value <> "" And InStr(cell.Text, "Requesting") = 0 Then
                ' Wait a bit more for all data
                Application.Wait (Now + TimeValue("0:00:02"))
                Exit Do
            End If
        End If
    Loop
End Sub

Private Sub EnsureDirectoryExists(path As String)
    ' Create directory if it doesn't exist
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")

    If Not fso.FolderExists(path) Then
        fso.CreateFolder path
    End If
End Sub

Private Sub SaveAsCSV(ws As Worksheet, filePath As String)
    ' Save worksheet as CSV
    Dim tempWb As Workbook
    Dim tempWs As Worksheet

    ' Copy to new workbook
    ws.Copy
    Set tempWb = ActiveWorkbook
    Set tempWs = tempWb.Sheets(1)

    ' Convert formulas to values
    tempWs.UsedRange.Value = tempWs.UsedRange.Value

    ' Save as CSV
    Application.DisplayAlerts = False
    tempWb.SaveAs Filename:=filePath, FileFormat:=xlCSV
    tempWb.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub

' ============ QUICK ACCESS MACROS ============

Public Sub QuickExtractOHLCV_AAPL()
    ' Quick test: extract just AAPL OHLCV
    Dim tickers As Variant
    tickers = Array("AAPL")

    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "AAPL_Test"

    ws.Range("A1").Formula = "=BDH(""AAPL US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""20240101"",""" & Format(Date, "YYYYMMDD") & """,""Dir=V"")"

    MsgBox "Formula inserted. Wait for data to populate, then check cell A1.", vbInformation
End Sub

Public Sub ShowBloombergStatus()
    ' Check if Bloomberg is connected
    On Error Resume Next

    Dim testVal As Variant
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets.Add

    ws.Range("A1").Formula = "=BDP(""SPY US Equity"",""PX_LAST"")"

    Application.Wait (Now + TimeValue("0:00:05"))

    testVal = ws.Range("A1").Value

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    If IsError(testVal) Or testVal = "" Or InStr(CStr(testVal), "N/A") > 0 Then
        MsgBox "Bloomberg connection: NOT WORKING" & vbCrLf & _
               "Check that Bloomberg Terminal is running and logged in.", vbExclamation
    Else
        MsgBox "Bloomberg connection: OK" & vbCrLf & _
               "SPY Last Price: " & testVal, vbInformation
    End If
End Sub
