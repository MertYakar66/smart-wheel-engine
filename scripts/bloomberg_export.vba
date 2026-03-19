' ============================================================================
' Bloomberg Historical Data Export Macro - SIMPLIFIED VERSION
' ============================================================================
'
' This version uses Windows Sleep API to truly yield control to Bloomberg
'
' USAGE:
'   1. Open VBA Editor (Alt+F11)
'   2. Import this module
'   3. Run: ExportSingleTicker (for testing) or ExportBloombergData (all tickers)
'
' ============================================================================

Option Explicit

' Windows API for true sleep (yields control unlike Application.Wait)
#If VBA7 Then
    Private Declare PtrSafe Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As LongPtr)
#Else
    Private Declare Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As Long)
#End If

' ============================================================================
' CONFIGURATION
' ============================================================================

Const OUTPUT_DIR As String = "C:\BloombergExport\"
Const START_DATE As String = "20240101"
Const END_DATE As String = "20260317"

' How long to wait for Bloomberg data (milliseconds)
' BDH with 2+ years of data needs ~30-60 seconds
Const BLOOMBERG_WAIT_MS As Long = 45000  ' 45 seconds

Function GetTickers() As Variant
    GetTickers = Array( _
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", _
        "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS", "PYPL", _
        "BAC", "NFLX", "ADBE", "CRM", "PFE", "TMO", "COST", "NKE" _
    )
End Function


' ============================================================================
' MAIN: Export Single Ticker (for testing)
' ============================================================================

Sub ExportSingleTicker()
    Dim ticker As String
    ticker = InputBox("Enter ticker symbol:", "Export Single Ticker", "AAPL")
    If Len(ticker) = 0 Then Exit Sub

    Debug.Print "=========================================="
    Debug.Print "Exporting: " & ticker & " at " & Now()
    Debug.Print "=========================================="

    ' Verify output directory
    If Dir(OUTPUT_DIR, vbDirectory) = "" Then
        MsgBox "Create folder first: " & OUTPUT_DIR, vbCritical
        Exit Sub
    End If

    ExportTickerData ticker

    MsgBox "Export complete for " & ticker & "!" & vbCrLf & _
           "Check: " & OUTPUT_DIR, vbInformation
End Sub


' ============================================================================
' MAIN: Export All Tickers
' ============================================================================

Sub ExportBloombergData()
    Dim tickers As Variant
    Dim i As Long

    tickers = GetTickers()

    Debug.Print "=========================================="
    Debug.Print "Bloomberg Export Started: " & Now()
    Debug.Print "Tickers: " & UBound(tickers) + 1
    Debug.Print "=========================================="

    ' Verify output directory
    If Dir(OUTPUT_DIR, vbDirectory) = "" Then
        MsgBox "Create folder first: " & OUTPUT_DIR, vbCritical
        Exit Sub
    End If

    For i = LBound(tickers) To UBound(tickers)
        Debug.Print vbCrLf & "--- " & (i + 1) & "/" & (UBound(tickers) + 1) & ": " & tickers(i) & " ---"
        ExportTickerData CStr(tickers(i))
    Next i

    Debug.Print vbCrLf & "=========================================="
    Debug.Print "ALL EXPORTS COMPLETE: " & Now()
    Debug.Print "=========================================="

    MsgBox "All exports complete!", vbInformation
End Sub


' ============================================================================
' CORE: Export all data types for one ticker
' ============================================================================

Sub ExportTickerData(ticker As String)
    ' OHLCV
    ExportBDH ticker, "ohlcv", _
        "PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME", _
        Array("Date", "Open", "High", "Low", "Close", "Volume")

    ' IV History
    ExportBDH ticker, "iv", _
        "30DAY_IMPVOL_100.0%MNY_DF", _
        Array("Date", "IV_30D")

    ' Earnings
    ExportBDH ticker, "earnings", _
        "IS_EPS,BEST_EPS", _
        Array("Date", "EPS_Actual", "EPS_Estimate"), "Period=Q"

    ' Dividends
    ExportBDH ticker, "dividends", _
        "EQY_DVD_YLD_IND", _
        Array("Date", "Dividend_Yield")
End Sub


' ============================================================================
' CORE: Export a single BDH request
' ============================================================================

Sub ExportBDH(ticker As String, dataType As String, fields As String, _
              headers As Variant, Optional extraParams As String = "")

    Dim ws As Worksheet
    Dim formula As String
    Dim outputPath As String
    Dim dataRange As Range
    Dim lastRow As Long, lastCol As Long

    On Error GoTo ErrHandler

    ' Create temp worksheet
    Set ws = ThisWorkbook.Worksheets.Add
    ws.Name = Left(ticker & "_" & dataType & "_" & Format(Now(), "hhmmss"), 31)

    ' Build and insert formula
    formula = "=BDH(""" & ticker & " US Equity"",""" & fields & _
              """,""" & START_DATE & """,""" & END_DATE & """,""Dir=V"""
    If Len(extraParams) > 0 Then formula = formula & ",""" & extraParams & """"
    formula = formula & ")"

    Debug.Print "  " & dataType & ": inserting formula..."
    ws.Range("A1").Formula = formula

    ' THE KEY: Use Windows Sleep to truly yield control to Bloomberg
    Debug.Print "  " & dataType & ": waiting " & (BLOOMBERG_WAIT_MS / 1000) & "s for Bloomberg..."
    DoEvents
    Sleep BLOOMBERG_WAIT_MS
    DoEvents

    ' Check if data arrived
    If InStr(CStr(ws.Range("A1").Value), "Requesting") > 0 Or _
       InStr(CStr(ws.Range("A1").Value), "#N/A") > 0 Then
        Debug.Print "  [FAIL] " & dataType & " - still requesting or error: " & ws.Range("A1").Value
        CleanupSheet ws
        Exit Sub
    End If

    ' Find data range
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    If lastRow < 2 Then
        Debug.Print "  [FAIL] " & dataType & " - no data rows"
        CleanupSheet ws
        Exit Sub
    End If

    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol))

    ' Convert to values (break Bloomberg link)
    dataRange.Copy
    dataRange.PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    ' Add headers
    ws.Rows(1).Insert
    Dim h As Long
    For h = LBound(headers) To UBound(headers)
        ws.Cells(1, h + 1).Value = headers(h)
    Next h

    ' Recalculate range with header
    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow + 1, lastCol))

    ' Save as CSV
    outputPath = OUTPUT_DIR & ticker & "_" & dataType & ".csv"
    SaveRangeAsCSV dataRange, outputPath

    Debug.Print "  [OK] " & outputPath & " (" & lastRow & " rows)"

    CleanupSheet ws
    Exit Sub

ErrHandler:
    Debug.Print "  [ERROR] " & dataType & ": " & Err.Description
    On Error Resume Next
    CleanupSheet ws
End Sub


' ============================================================================
' HELPER: Save range to CSV
' ============================================================================

Sub SaveRangeAsCSV(rng As Range, filePath As String)
    Dim tempWb As Workbook

    rng.Copy
    Set tempWb = Workbooks.Add
    tempWb.Sheets(1).Range("A1").PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    Application.DisplayAlerts = False
    tempWb.SaveAs Filename:=filePath, FileFormat:=xlCSV
    tempWb.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub


' ============================================================================
' HELPER: Delete worksheet
' ============================================================================

Sub CleanupSheet(ws As Worksheet)
    On Error Resume Next
    Application.DisplayAlerts = False
    ws.Delete
    Application.DisplayAlerts = True
    On Error GoTo 0
End Sub


' ============================================================================
' TEST: Quick Bloomberg connection test
' ============================================================================

Sub TestBloomberg()
    Dim ws As Worksheet

    Set ws = ThisWorkbook.Worksheets.Add
    ws.Range("A1").Formula = "=BDP(""AAPL US Equity"",""PX_LAST"")"

    Debug.Print "Testing Bloomberg connection..."
    DoEvents
    Sleep 5000  ' 5 seconds should be enough for BDP
    DoEvents

    If IsNumeric(ws.Range("A1").Value) Then
        MsgBox "Bloomberg OK! AAPL = " & ws.Range("A1").Value, vbInformation
    Else
        MsgBox "Bloomberg FAILED: " & ws.Range("A1").Value, vbCritical
    End If

    CleanupSheet ws
End Sub
