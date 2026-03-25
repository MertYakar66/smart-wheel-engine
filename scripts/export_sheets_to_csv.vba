' ============================================================================
' Export All Sheets to CSV - Run AFTER data has loaded
' ============================================================================
'
' USAGE:
'   1. Create sheets manually, paste one BDH formula per sheet
'   2. Name each sheet with the ticker (e.g., "AAPL", "MSFT")
'   3. Wait for ALL data to load
'   4. Run this macro to export all sheets to CSV
'
' ============================================================================

Option Explicit

Const OUTPUT_DIR As String = "C:\BloombergExport\ohlcv\"

Sub ExportAllSheetsToCsv()
    Dim ws As Worksheet
    Dim outputPath As String
    Dim count As Long

    ' Create output directory if needed
    On Error Resume Next
    MkDir "C:\BloombergExport"
    MkDir OUTPUT_DIR
    On Error GoTo 0

    count = 0

    For Each ws In ThisWorkbook.Worksheets
        ' Skip sheets that don't look like tickers
        If Len(ws.Name) <= 5 And ws.Name <> "Sheet1" Then
            ' Check if there's data
            If ws.Range("A1").Value <> "" And _
               Not IsError(ws.Range("A1").Value) And _
               InStr(CStr(ws.Range("A1").Value), "Requesting") = 0 Then

                outputPath = OUTPUT_DIR & ws.Name & "_ohlcv.csv"
                ExportSheetToCsv ws, outputPath
                count = count + 1
                Debug.Print "Exported: " & ws.Name
            Else
                Debug.Print "Skipped (no data): " & ws.Name
            End If
        End If
    Next ws

    MsgBox "Exported " & count & " sheets to:" & vbCrLf & OUTPUT_DIR, vbInformation
End Sub


Sub ExportSheetToCsv(ws As Worksheet, filePath As String)
    Dim lastRow As Long, lastCol As Long
    Dim dataRange As Range
    Dim tempWb As Workbook

    ' Find data range
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    If lastRow < 2 Then Exit Sub

    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol))

    ' Copy to temp workbook and save as CSV
    dataRange.Copy
    Set tempWb = Workbooks.Add
    tempWb.Sheets(1).Range("A1").PasteSpecial xlPasteValues
    Application.CutCopyMode = False

    ' Add headers
    tempWb.Sheets(1).Rows(1).Insert
    tempWb.Sheets(1).Range("A1:F1").Value = Array("Date", "Open", "High", "Low", "Close", "Volume")

    Application.DisplayAlerts = False
    tempWb.SaveAs Filename:=filePath, FileFormat:=xlCSV
    tempWb.Close SaveChanges:=False
    Application.DisplayAlerts = True
End Sub


' ============================================================================
' Helper: Create 50 empty sheets named by ticker
' ============================================================================

Sub CreateTickerSheets()
    Dim tickers As Variant
    Dim i As Long
    Dim ws As Worksheet

    ' First 50 tickers
    tickers = Array("MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", _
                    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", _
                    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", _
                    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT", _
                    "APP", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK")

    Application.ScreenUpdating = False

    For i = LBound(tickers) To UBound(tickers)
        Set ws = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        ws.Name = tickers(i)
    Next i

    Application.ScreenUpdating = True

    MsgBox "Created " & (UBound(tickers) + 1) & " sheets. Now paste formulas!", vbInformation
End Sub


' ============================================================================
' Helper: Paste formula to ALL ticker sheets at once
' ============================================================================

Sub PasteFormulaToAllSheets()
    Dim ws As Worksheet
    Dim formula As String

    Application.ScreenUpdating = False

    For Each ws In ThisWorkbook.Worksheets
        If Len(ws.Name) <= 5 And ws.Name <> "Sheet1" Then
            formula = "=BDH(""" & ws.Name & " US Equity"",""PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME"",""20240101"",""20260317"",""Dir=V"")"
            ws.Range("A1").Formula = formula
        End If
    Next ws

    Application.ScreenUpdating = True

    MsgBox "Formulas pasted! Wait for all data to load, then run ExportAllSheetsToCsv", vbInformation
End Sub
