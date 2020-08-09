from PyPDF2.generic import (
    DictionaryObject,
    NumberObject,
    FloatObject,
    NameObject,
    TextStringObject,
    ArrayObject,
    BooleanObject
)

# x1, y1 starts in bottom left corner
def createHighlight(x1, y1, x2, y2, meta):
    color = [255.0 / 255.0, 209 / 255.0, 0]
    newHighlight = DictionaryObject()
    # https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf
    newHighlight.update({
        # Table 165 NoZoom
        NameObject("/F"): NumberObject(4),
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Text"),

        # Table 170 titlebar
        NameObject("/T"): TextStringObject(meta["author"]),
        NameObject("/Contents"): TextStringObject(meta["contents"]),
        
        # Table 164 color, annotation rectangle
        NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
        NameObject("/Rect"): ArrayObject([
            FloatObject(x1),
            FloatObject(y1),
            FloatObject(x2),
            FloatObject(y2)
        ]),
        # 12.5.6.4 text annotation
        NameObject('/Open'): BooleanObject(True),
        NameObject('/Name'): NameObject('/Comment'),

        # NameObject("/QuadPoints"): ArrayObject([
        #     FloatObject(x1),
        #     FloatObject(y2),
        #     FloatObject(x2),
        #     FloatObject(y2),
        #     FloatObject(x1),
        #     FloatObject(y1),
        #     FloatObject(x2),
        #     FloatObject(y1)
        # ]),
    })

    # newHighlight.update({
    #     NameObject("/F"): NumberObject(4),
    #     NameObject("/Type"): NameObject("/Annot"),
    #     NameObject("/Subtype"): NameObject("/Highlight"),

    #     NameObject("/T"): TextStringObject(meta["author"]),
    #     NameObject("/Contents"): TextStringObject(meta["contents"]),

    #     NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
    #     NameObject("/Rect"): ArrayObject([
    #         FloatObject(x1),
    #         FloatObject(y1),
    #         FloatObject(x2),
    #         FloatObject(y2)
    #     ]),
    #     NameObject("/QuadPoints"): ArrayObject([
    #         FloatObject(x1),
    #         FloatObject(y2),
    #         FloatObject(x2),
    #         FloatObject(y2),
    #         FloatObject(x1),
    #         FloatObject(y1),
    #         FloatObject(x2),
    #         FloatObject(y1)
    #     ]),
    # })

    return newHighlight

def addHighlightToPage(highlight, page, output):
    highlight_ref = output._addObject(highlight);

    if "/Annots" in page:
        page[NameObject("/Annots")].append(highlight_ref)
    else:
        page[NameObject("/Annots")] = ArrayObject([highlight_ref])

from PyPDF2 import PdfFileWriter, PdfFileReader

pdfInput = PdfFileReader(open("../data/yasser.pdf", "rb"))
pdfOutput = PdfFileWriter()

page1 = pdfInput.getPage(0)
print(page1.mediaBox)
print(page1.mediaBox[0])

highlight = createHighlight(504 / 2, 0, 0, 792 / 2, {
    "author": "",
    "contents": "Bla-bla-bla"
})

addHighlightToPage(highlight, page1, pdfOutput)

pdfOutput.addPage(page1)

outputStream = open("../data/yasser2.pdf", "wb")
pdfOutput.write(outputStream)