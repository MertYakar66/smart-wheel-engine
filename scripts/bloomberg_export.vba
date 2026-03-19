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

' ============================================================================
' CONFIGURATION
' ============================================================================

Const OUTPUT_DIR As String = "C:\BloombergExport\"
Const START_DATE As String = "20240101"
Const END_DATE As String = "20260317"

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
' Uses batch approach: insert all formulas, wait once, then export all
' ============================================================================

Sub ExportTickerData(ticker As String)
    Dim wsOHLCV As Worksheet, wsIV As Worksheet
    Dim wsEarnings As Worksheet, wsDividends As Worksheet
    Dim ts As String

    ts = Format(Now(), "hhmmss")

    ' Create all sheets and insert formulas at once
    Set wsOHLCV = ThisWorkbook.Worksheets.Add
    wsOHLCV.Name = Left(ticker & "_ohlcv_" & ts, 31)
    wsOHLCV.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""" & START_DATE & """,""" & END_DATE & """,""Dir=V"")"

    Set wsIV = ThisWorkbook.Worksheets.Add
    wsIV.Name = Left(ticker & "_iv_" & ts, 31)
    wsIV.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""30DAY_IMPVOL_100.0%MNY_DF"",""" & START_DATE & """,""" & END_DATE & """,""Dir=V"")"

    Set wsEarnings = ThisWorkbook.Worksheets.Add
    wsEarnings.Name = Left(ticker & "_earn_" & ts, 31)
    wsEarnings.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""IS_EPS,BEST_EPS"",""" & START_DATE & """,""" & END_DATE & """,""Dir=V"",""Period=Q"")"

    Set wsDividends = ThisWorkbook.Worksheets.Add
    wsDividends.Name = Left(ticker & "_div_" & ts, 31)
    wsDividends.Range("A1").Formula = "=BDH(""" & ticker & " US Equity"",""EQY_DVD_YLD_IND"",""" & START_DATE & """,""" & END_DATE & """,""Dir=V"")"

    ' Single wait for all 4 to load
    MsgBox "Wait for ALL 4 sheets to show data (check each tab), then click OK." & vbCrLf & vbCrLf & _
           "Sheets: " & wsOHLCV.Name & ", " & wsIV.Name & ", " & wsEarnings.Name & ", " & wsDividends.Name, _
           vbInformation, ticker & " - Wait for Bloomberg"

    ' Now export each
    ExportSheetToCSV wsOHLCV, ticker, "ohlcv", Array("Date", "Open", "High", "Low", "Close", "Volume")
    ExportSheetToCSV wsIV, ticker, "iv", Array("Date", "IV_30D")
    ExportSheetToCSV wsEarnings, ticker, "earnings", Array("Date", "EPS_Actual", "EPS_Estimate")
    ExportSheetToCSV wsDividends, ticker, "dividends", Array("Date", "Dividend_Yield")
End Sub


' ============================================================================
' HELPER: Export a sheet that already has data to CSV
' ============================================================================

Sub ExportSheetToCSV(ws As Worksheet, ticker As String, dataType As String, headers As Variant)
    Dim lastRow As Long, lastCol As Long
    Dim dataRange As Range
    Dim outputPath As String
    Dim h As Long

    On Error GoTo ErrHandler

    ' Check if data loaded
    If InStr(CStr(ws.Range("A1").Value), "Requesting") > 0 Or _
       InStr(CStr(ws.Range("A1").Value), "#N/A") > 0 Then
        Debug.Print "  [SKIP] " & dataType & " - no data: " & ws.Range("A1").Value
        CleanupSheet ws
        Exit Sub
    End If

    ' Find data range
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    If lastRow < 2 Then
        Debug.Print "  [SKIP] " & dataType & " - no data rows"
        CleanupSheet ws
        Exit Sub
    End If

    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol))

    ' Convert to values
    dataRange.Copy
    dataRange.PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    ' Add headers
    ws.Rows(1).Insert
    For h = LBound(headers) To UBound(headers)
        ws.Cells(1, h + 1).Value = headers(h)
    Next h

    ' Recalculate range
    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow + 1, lastCol))

    ' Save
    outputPath = OUTPUT_DIR & ticker & "_" & dataType & ".csv"
    SaveRangeAsCSV dataRange, outputPath
    Debug.Print "  [OK] " & outputPath & " (" & lastRow & " rows)"

    CleanupSheet ws
    Exit Sub

ErrHandler:
    Debug.Print "  [ERROR] " & dataType & ": " & Err.Description
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

    ' MsgBox truly yields control - click OK when you see data in A1
    MsgBox "Look at the sheet - wait until A1 shows a price, then click OK", vbInformation

    If IsNumeric(ws.Range("A1").Value) Then
        MsgBox "Bloomberg OK! AAPL = " & ws.Range("A1").Value, vbInformation
    Else
        MsgBox "Bloomberg FAILED: " & ws.Range("A1").Value, vbCritical
    End If

    CleanupSheet ws
End Sub
