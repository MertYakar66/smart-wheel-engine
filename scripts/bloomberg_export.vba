' ============================================================================
' Bloomberg Historical Data Export Macro
' ============================================================================
'
' USAGE:
'   1. Open this file in VBA Editor (Alt+F11)
'   2. Import this module into your workbook
'   3. Run: ExportBloombergData
'
' REQUIREMENTS:
'   - Bloomberg Terminal must be running and logged in
'   - Bloomberg Excel Add-in must be enabled
'   - Connection verified: =BDP("AAPL US Equity","PX_LAST") should return a value
'
' OUTPUT:
'   CSVs saved to: C:\BloombergExport\ (configurable below)
'
' DATA CATEGORIES EXPORTED:
'   - OHLCV: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME
'   - IV History: 30DAY_IMPVOL_100.0%MNY_DF
'   - Earnings: IS_EPS, BEST_EPS (quarterly)
'   - Dividend Yield Proxy: EQY_DVD_YLD_IND
'
' NOTE: DVD_EX_DT does not work via BDH() - using yield proxy instead
' ============================================================================

Option Explicit

' ============================================================================
' CONFIGURATION
' ============================================================================

' Output directory (create this folder before running)
Const OUTPUT_DIR As String = "C:\BloombergExport\"

' Date range for historical data
Const START_DATE As String = "20240101"
Const END_DATE As String = "20260317"

' Maximum wait time for Bloomberg data (seconds)
Const MAX_WAIT_SECONDS As Long = 60

' Check interval (seconds)
Const CHECK_INTERVAL As Double = 0.5

' Tickers to export (modify as needed)
Function GetTickers() As Variant
    GetTickers = Array( _
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", _
        "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS", "PYPL", _
        "BAC", "NFLX", "ADBE", "CRM", "PFE", "TMO", "COST", "NKE" _
    )
End Function


' ============================================================================
' MAIN ENTRY POINT
' ============================================================================

Sub ExportBloombergData()
    Dim tickers As Variant
    Dim ticker As Variant
    Dim successCount As Long
    Dim failCount As Long

    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    tickers = GetTickers()
    successCount = 0
    failCount = 0

    Debug.Print "=========================================="
    Debug.Print "Bloomberg Export Started: " & Now()
    Debug.Print "Output Directory: " & OUTPUT_DIR
    Debug.Print "Tickers: " & UBound(tickers) + 1
    Debug.Print "=========================================="

    ' Verify output directory exists
    If Dir(OUTPUT_DIR, vbDirectory) = "" Then
        MsgBox "Output directory does not exist: " & OUTPUT_DIR & vbCrLf & _
               "Please create it before running.", vbCritical
        Exit Sub
    End If

    ' Export each ticker
    For Each ticker In tickers
        Debug.Print vbCrLf & "Processing: " & ticker

        ' Export OHLCV
        If ExportSingleRequest(CStr(ticker), "ohlcv", _
            "PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME", START_DATE, END_DATE, "") Then
            successCount = successCount + 1
        Else
            failCount = failCount + 1
        End If

        ' Export IV History
        If ExportSingleRequest(CStr(ticker), "iv", _
            "30DAY_IMPVOL_100.0%MNY_DF", START_DATE, END_DATE, "") Then
            successCount = successCount + 1
        Else
            failCount = failCount + 1
        End If

        ' Export Earnings (quarterly)
        If ExportSingleRequest(CStr(ticker), "earnings", _
            "IS_EPS,BEST_EPS", START_DATE, END_DATE, "Period=Q") Then
            successCount = successCount + 1
        Else
            failCount = failCount + 1
        End If

        ' Export Dividend Yield Proxy
        If ExportSingleRequest(CStr(ticker), "dividends", _
            "EQY_DVD_YLD_IND", START_DATE, END_DATE, "") Then
            successCount = successCount + 1
        Else
            failCount = failCount + 1
        End If

        DoEvents
    Next ticker

    Application.ScreenUpdating = True
    Application.DisplayAlerts = True

    Debug.Print vbCrLf & "=========================================="
    Debug.Print "Export Complete: " & Now()
    Debug.Print "Success: " & successCount & ", Failed: " & failCount
    Debug.Print "=========================================="

    MsgBox "Export Complete!" & vbCrLf & _
           "Success: " & successCount & vbCrLf & _
           "Failed: " & failCount, vbInformation
End Sub


' ============================================================================
' CORE EXPORT FUNCTION - One Request at a Time with Async Wait
' ============================================================================

Function ExportSingleRequest(ticker As String, dataType As String, _
    fields As String, startDate As String, endDate As String, _
    Optional extraParams As String = "") As Boolean

    Dim ws As Worksheet
    Dim formula As String
    Dim outputPath As String
    Dim waitResult As Boolean
    Dim dataRange As Range

    On Error GoTo ErrorHandler

    ' Create fresh temporary worksheet
    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "TempExport_" & Format(Now(), "hhmmss")

    ' Build BDH formula
    formula = BuildBDHFormula(ticker, fields, startDate, endDate, extraParams)
    Debug.Print "  Formula: " & Left(formula, 80) & "..."

    ' Insert formula in A1
    ws.Range("A1").Formula2 = formula

    ' Force calculation
    Application.Calculate

    ' Wait for Bloomberg to resolve
    waitResult = WaitForBloombergData(ws.Range("A1"))

    If Not waitResult Then
        Debug.Print "  [FAIL] " & ticker & "_" & dataType & " - Timeout or error"
        CleanupWorksheet ws
        ExportSingleRequest = False
        Exit Function
    End If

    ' Determine data range (spilled array)
    Set dataRange = GetSpilledRange(ws.Range("A1"))

    If dataRange Is Nothing Or dataRange.Rows.Count < 2 Then
        Debug.Print "  [FAIL] " & ticker & "_" & dataType & " - No data returned"
        CleanupWorksheet ws
        ExportSingleRequest = False
        Exit Function
    End If

    ' Copy values (breaks link to Bloomberg)
    dataRange.Copy
    dataRange.PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    ' Export to CSV
    outputPath = OUTPUT_DIR & ticker & "_" & dataType & ".csv"
    ExportRangeToCSV dataRange, outputPath

    Debug.Print "  [OK] " & outputPath & " (" & dataRange.Rows.Count & " rows)"

    ' Cleanup
    CleanupWorksheet ws
    ExportSingleRequest = True
    Exit Function

ErrorHandler:
    Debug.Print "  [ERROR] " & ticker & "_" & dataType & ": " & Err.Description
    On Error Resume Next
    CleanupWorksheet ws
    ExportSingleRequest = False
End Function


' ============================================================================
' BLOOMBERG ASYNC WAIT - The Critical Fix
' ============================================================================

Function WaitForBloombergData(cell As Range) As Boolean
    Dim startTime As Double
    Dim cellValue As String
    Dim elapsed As Double

    startTime = Timer

    Do
        DoEvents
        Application.Calculate

        cellValue = CStr(cell.Value)

        ' Check for success (not requesting, not error)
        If InStr(cellValue, "Requesting") = 0 And _
           InStr(cellValue, "#N/A") = 0 And _
           Len(Trim(cellValue)) > 0 Then
            WaitForBloombergData = True
            Exit Function
        End If

        ' Check for permanent error
        If InStr(cellValue, "#N/A Field Not Applicable") > 0 Or _
           InStr(cellValue, "#N/A Invalid") > 0 Or _
           InStr(cellValue, "#N/A N/A") > 0 Then
            ' Some #N/A are permanent errors
            If InStr(cellValue, "Requesting") = 0 Then
                WaitForBloombergData = False
                Exit Function
            End If
        End If

        ' Check timeout
        elapsed = Timer - startTime
        If elapsed < 0 Then elapsed = elapsed + 86400  ' Handle midnight

        If elapsed > MAX_WAIT_SECONDS Then
            Debug.Print "  Timeout after " & MAX_WAIT_SECONDS & "s"
            WaitForBloombergData = False
            Exit Function
        End If

        ' Wait before next check
        Application.Wait Now + TimeSerial(0, 0, CHECK_INTERVAL)

    Loop
End Function


' ============================================================================
' HELPER FUNCTIONS
' ============================================================================

Function BuildBDHFormula(ticker As String, fields As String, _
    startDate As String, endDate As String, _
    Optional extraParams As String = "") As String

    Dim formula As String

    formula = "=BDH(""" & ticker & " US Equity"",""" & fields & _
              """,""" & startDate & """,""" & endDate & """,""Dir=V"""

    If Len(extraParams) > 0 Then
        formula = formula & ",""" & extraParams & """"
    End If

    formula = formula & ")"

    BuildBDHFormula = formula
End Function


Function GetSpilledRange(startCell As Range) As Range
    ' Get the full spilled range from a dynamic array formula
    On Error Resume Next
    Set GetSpilledRange = startCell.SpillingToRange

    ' Fallback: detect used range manually
    If GetSpilledRange Is Nothing Then
        Dim lastRow As Long, lastCol As Long
        With startCell.Worksheet
            lastRow = .Cells(.Rows.Count, startCell.Column).End(xlUp).Row
            lastCol = .Cells(startCell.Row, .Columns.Count).End(xlToLeft).Column
            If lastRow >= startCell.Row And lastCol >= startCell.Column Then
                Set GetSpilledRange = .Range(startCell, .Cells(lastRow, lastCol))
            End If
        End With
    End If
    On Error GoTo 0
End Function


Sub ExportRangeToCSV(rng As Range, filePath As String)
    Dim ws As Worksheet
    Dim tempWb As Workbook

    ' Copy to new workbook for clean CSV export
    rng.Copy
    Set tempWb = Workbooks.Add
    tempWb.Sheets(1).Range("A1").PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    ' Save as CSV
    Application.DisplayAlerts = False
    tempWb.SaveAs Filename:=filePath, FileFormat:=xlCSV
    tempWb.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub


Sub CleanupWorksheet(ws As Worksheet)
    On Error Resume Next
    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True
    On Error GoTo 0
End Sub


' ============================================================================
' UTILITY: Test Bloomberg Connection
' ============================================================================

Sub TestBloombergConnection()
    Dim ws As Worksheet
    Dim testResult As String

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = "BBG_Test_" & Format(Now(), "hhmmss")

    ' Simple BDP test
    ws.Range("A1").Formula = "=BDP(""AAPL US Equity"",""PX_LAST"")"
    Application.Calculate

    ' Wait for result
    If WaitForBloombergData(ws.Range("A1")) Then
        testResult = "Bloomberg Connection OK: AAPL = " & ws.Range("A1").Value
        MsgBox testResult, vbInformation
    Else
        testResult = "Bloomberg Connection FAILED: " & ws.Range("A1").Value
        MsgBox testResult, vbCritical
    End If

    Debug.Print testResult
    CleanupWorksheet ws
End Sub


' ============================================================================
' UTILITY: Export Single Ticker (for testing)
' ============================================================================

Sub ExportSingleTicker()
    Dim ticker As String
    ticker = InputBox("Enter ticker symbol:", "Export Single Ticker", "AAPL")

    If Len(ticker) = 0 Then Exit Sub

    Application.ScreenUpdating = False

    ExportSingleRequest ticker, "ohlcv", _
        "PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME", START_DATE, END_DATE, ""

    ExportSingleRequest ticker, "iv", _
        "30DAY_IMPVOL_100.0%MNY_DF", START_DATE, END_DATE, ""

    ExportSingleRequest ticker, "earnings", _
        "IS_EPS,BEST_EPS", START_DATE, END_DATE, "Period=Q"

    ExportSingleRequest ticker, "dividends", _
        "EQY_DVD_YLD_IND", START_DATE, END_DATE, ""

    Application.ScreenUpdating = True

    MsgBox "Export complete for " & ticker, vbInformation
End Sub
