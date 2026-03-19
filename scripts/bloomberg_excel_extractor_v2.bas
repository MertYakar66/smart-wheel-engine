' Bloomberg Data Extractor for Smart Wheel Engine - V2
' ================================================
' FIXED VERSION with longer wait times for Bloomberg data
'
' SETUP:
' 1. Open Excel with Bloomberg Add-in loaded
' 2. Press Alt+F11 to open VBA Editor
' 3. Delete old module, Insert > Module
' 4. Paste this entire file
' 5. Close VBA Editor (Alt+Q)
' 6. Run macros from Alt+F8

Option Explicit

' ============ CONFIGURATION ============
Private Const BLOOMBERG_DIR As String = "C:\Users\mertmert\Desktop\smart-wheel-engine\data\bloomberg\"
Private Const START_DATE As String = "20190101"

' Your tickers
Private Function GetTickers() As Variant
    GetTickers = Array("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", _
                       "JPM", "BAC", "WFC", "GS", _
                       "UNH", "JNJ", "LLY", "ABBV", _
                       "XOM", "CVX")
End Function

' ============ MAIN EXTRACTION ROUTINES ============

Public Sub ExtractAllData()
    Dim startTime As Double
    startTime = Timer

    MsgBox "Starting full data extraction. This may take 60-90 minutes." & vbCrLf & _
           "Do not touch Excel while running.", vbInformation

    ExtractOHLCV
    ExtractIVHistory
    ExtractEarnings
    ExtractDividends
    ExtractRates
    ExtractFundamentals

    MsgBox "Extraction complete! Time: " & Format((Timer - startTime) / 60, "0.0") & " minutes", vbInformation
End Sub

Public Sub ExtractOHLCV()
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String
    Dim i As Integer

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationAutomatic

    i = 0
    For Each ticker In tickers
        i = i + 1
        Application.StatusBar = "Extracting OHLCV " & i & "/" & UBound(tickers) + 1 & ": " & ticker

        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_OHLCV", 31)

        ' BDH formula for OHLCV
        ws.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""" & START_DATE & """,""" & endDate & """,""Dir=V"")"

        ' Wait for data - LONGER TIMEOUT
        WaitForBloombergData ws.Range("A1"), 180

        ' Extra wait to ensure all data loaded
        Application.Wait (Now + TimeValue("0:00:05"))

        ' Check if we got data
        If IsEmpty(ws.Range("B2").Value) Or ws.Range("B2").Value = "" Then
            Debug.Print "WARNING: No price data for " & ticker
        End If

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

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "OHLCV extraction complete! " & UBound(tickers) + 1 & " tickers saved.", vbInformation
End Sub

Public Sub ExtractIVHistory()
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String
    Dim formula As String
    Dim i As Integer

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    i = 0
    For Each ticker In tickers
        i = i + 1
        Application.StatusBar = "Extracting IV History " & i & "/" & UBound(tickers) + 1 & ": " & ticker

        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_IV", 31)

        ' BDH formula for IV metrics
        formula = "=BDH(""" & ticker & " US Equity""," & _
                  """30DAY_IMPVOL_100.0%MNY_DF,60DAY_IMPVOL_100.0%MNY_DF," & _
                  "30DAY_IMPVOL_90.0%MNY_DF,30DAY_IMPVOL_110.0%MNY_DF," & _
                  "20DAY_HV,60DAY_HV""," & _
                  """" & START_DATE & """,""" & endDate & """,""Dir=V"")"

        ws.Range("A1").Formula = formula

        WaitForBloombergData ws.Range("A1"), 180
        Application.Wait (Now + TimeValue("0:00:05"))

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

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "IV history extraction complete!", vbInformation
End Sub

Public Sub ExtractEarnings()
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String
    Dim i As Integer

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    i = 0
    For Each ticker In tickers
        i = i + 1
        Application.StatusBar = "Extracting Earnings " & i & "/" & UBound(tickers) + 1 & ": " & ticker

        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_ERN", 31)

        ' BDH for quarterly EPS data
        ws.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""IS_EPS,BEST_EPS_MEDIAN,EARN_EST_EPS_SURPRISE_PCT"",""" & START_DATE & """,""" & endDate & """,""Dir=V"",""Per=Q"")"

        WaitForBloombergData ws.Range("A1"), 120
        Application.Wait (Now + TimeValue("0:00:03"))

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

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "Earnings extraction complete!", vbInformation
End Sub

Public Sub ExtractDividends()
    Dim tickers As Variant
    Dim ticker As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String
    Dim i As Integer

    tickers = GetTickers()
    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False

    i = 0
    For Each ticker In tickers
        i = i + 1
        Application.StatusBar = "Extracting Dividends " & i & "/" & UBound(tickers) + 1 & ": " & ticker

        Set ws = ThisWorkbook.Worksheets.Add
        ws.Name = Left(CStr(ticker) & "_DIV", 31)

        ' BDS for dividend history
        ws.Range("A1").Formula = "=BDS(""" & ticker & " US Equity"",""DVD_HIST_ALL"",""DVD_START_DT=" & START_DATE & """,""DVD_END_DT=" & endDate & """)"

        WaitForBloombergData ws.Range("A1"), 120
        Application.Wait (Now + TimeValue("0:00:03"))

        outputPath = BLOOMBERG_DIR & "dividends\" & ticker & ".csv"
        EnsureDirectoryExists BLOOMBERG_DIR & "dividends\"
        SaveAsCSV ws, outputPath

        Application.DisplayAlerts = False
        ws.Delete
        Application.DisplayAlerts = True

        Debug.Print "Extracted dividends for " & ticker
    Next ticker

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "Dividend extraction complete!", vbInformation
End Sub

Public Sub ExtractRates()
    Dim ws As Worksheet
    Dim outputPath As String
    Dim endDate As String

    endDate = Format(Date, "YYYYMMDD")

    Application.ScreenUpdating = False
    Application.StatusBar = "Extracting Treasury Rates..."

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "Treasury_Yields"

    ' 3-month T-bill
    ws.Range("A1").Formula = "=BDH(""USGG3M Index"",""PX_LAST"",""" & START_DATE & """,""" & endDate & """,""Dir=V"")"

    WaitForBloombergData ws.Range("A1"), 180
    Application.Wait (Now + TimeValue("0:00:05"))

    ws.Range("A1").EntireRow.Insert
    ws.Range("A1:B1").Value = Array("Date", "Rate_3M")

    outputPath = BLOOMBERG_DIR & "rates\treasury_yields.csv"
    EnsureDirectoryExists BLOOMBERG_DIR & "rates\"
    SaveAsCSV ws, outputPath

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "Rates extraction complete!", vbInformation
End Sub

Public Sub ExtractFundamentals()
    Dim tickers As Variant
    Dim ws As Worksheet
    Dim outputPath As String
    Dim i As Long

    tickers = GetTickers()

    Application.ScreenUpdating = False
    Application.StatusBar = "Extracting Fundamentals..."

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "Fundamentals"

    ' Headers
    ws.Range("A1:G1").Value = Array("Ticker", "Market_Cap", "GICS_Sector", "GICS_Industry", "Div_Yield", "PE", "EPS")

    ' Fill ticker column and formulas
    For i = LBound(tickers) To UBound(tickers)
        Dim row As Long
        row = i + 2
        ws.Cells(row, 1).Value = tickers(i) & " US Equity"
        ws.Cells(row, 2).Formula = "=BDP(A" & row & ",""CUR_MKT_CAP"")"
        ws.Cells(row, 3).Formula = "=BDP(A" & row & ",""GICS_SECTOR_NAME"")"
        ws.Cells(row, 4).Formula = "=BDP(A" & row & ",""GICS_INDUSTRY_GROUP_NAME"")"
        ws.Cells(row, 5).Formula = "=BDP(A" & row & ",""EQY_DVD_YLD_IND"")"
        ws.Cells(row, 6).Formula = "=BDP(A" & row & ",""PE_RATIO"")"
        ws.Cells(row, 7).Formula = "=BDP(A" & row & ",""TRAIL_12M_EPS"")"
    Next i

    WaitForBloombergData ws.Cells(2, 2), 120
    Application.Wait (Now + TimeValue("0:00:05"))

    outputPath = BLOOMBERG_DIR & "fundamentals\sp500_fundamentals.csv"
    EnsureDirectoryExists BLOOMBERG_DIR & "fundamentals\"
    SaveAsCSV ws, outputPath

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    Application.StatusBar = False
    Application.ScreenUpdating = True
    MsgBox "Fundamentals extraction complete!", vbInformation
End Sub

' ============ IMPROVED HELPER FUNCTIONS ============

Private Sub WaitForBloombergData(cell As Range, timeoutSeconds As Long)
    ' Wait for Bloomberg data to populate - IMPROVED VERSION
    Dim startTime As Double
    Dim cellValue As Variant
    Dim lastValue As Variant
    Dim stableCount As Integer

    startTime = Timer
    stableCount = 0
    lastValue = ""

    Do While Timer - startTime < timeoutSeconds
        DoEvents
        Application.Wait (Now + TimeValue("0:00:03"))  ' Check every 3 seconds

        On Error Resume Next
        cellValue = cell.Value
        On Error GoTo 0

        ' Check if data has arrived (not requesting, not error, not empty)
        If Not IsError(cellValue) Then
            If cellValue <> "" And InStr(CStr(cellValue), "Requesting") = 0 And InStr(CStr(cellValue), "#N/A") = 0 Then
                ' Data appeared - wait for it to stabilize
                If CStr(cellValue) = CStr(lastValue) Then
                    stableCount = stableCount + 1
                    If stableCount >= 2 Then
                        ' Data stable for 2 checks (6 seconds) - good to go
                        Application.Wait (Now + TimeValue("0:00:03"))
                        Exit Do
                    End If
                Else
                    stableCount = 0
                End If
                lastValue = cellValue
            End If
        End If
    Loop

    ' Final wait to ensure all columns loaded
    Application.Wait (Now + TimeValue("0:00:02"))
End Sub

Private Sub EnsureDirectoryExists(path As String)
    Dim fso As Object
    Dim folders As Variant
    Dim currentPath As String
    Dim i As Integer

    Set fso = CreateObject("Scripting.FileSystemObject")

    ' Handle nested directories
    folders = Split(path, "\")
    currentPath = folders(0)

    For i = 1 To UBound(folders)
        If folders(i) <> "" Then
            currentPath = currentPath & "\" & folders(i)
            If Not fso.FolderExists(currentPath) Then
                fso.CreateFolder currentPath
            End If
        End If
    Next i
End Sub

Private Sub SaveAsCSV(ws As Worksheet, filePath As String)
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
    tempWb.SaveAs Filename:=filePath, FileFormat:=xlCSV, CreateBackup:=False
    tempWb.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub

' ============ TEST MACROS ============

Public Sub TestSingleTicker()
    ' Test extraction for just AAPL
    Dim ws As Worksheet
    Dim endDate As String

    endDate = Format(Date, "YYYYMMDD")

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "AAPL_Test"

    ws.Range("A1").Formula = "=BDH(""AAPL US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""20240101"",""" & endDate & """,""Dir=V"")"

    MsgBox "Formula inserted in sheet 'AAPL_Test'." & vbCrLf & _
           "Watch the cells - data should fill in within 30 seconds." & vbCrLf & _
           "Once you see dates and prices, the extraction will work.", vbInformation
End Sub

Public Sub ShowBloombergStatus()
    Dim ws As Worksheet
    Dim testVal As Variant

    On Error Resume Next

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Range("A1").Formula = "=BDP(""SPY US Equity"",""PX_LAST"")"

    ' Wait longer
    Application.Wait (Now + TimeValue("0:00:10"))
    DoEvents
    Application.Wait (Now + TimeValue("0:00:05"))

    testVal = ws.Range("A1").Value

    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True

    If IsError(testVal) Or testVal = "" Or InStr(CStr(testVal), "N/A") > 0 Or InStr(CStr(testVal), "Requesting") > 0 Then
        MsgBox "Bloomberg connection: NOT READY" & vbCrLf & vbCrLf & _
               "Make sure:" & vbCrLf & _
               "1. Bloomberg Terminal is running" & vbCrLf & _
               "2. You are logged in" & vbCrLf & _
               "3. Try a manual =BDP formula first", vbExclamation
    Else
        MsgBox "Bloomberg connection: OK" & vbCrLf & _
               "SPY Last Price: $" & Format(testVal, "0.00"), vbInformation
    End If
End Sub
