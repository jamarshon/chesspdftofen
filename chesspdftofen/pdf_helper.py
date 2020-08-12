from PyPDF2.generic import (
  DictionaryObject,
  NumberObject,
  FloatObject,
  NameObject,
  TextStringObject,
  ArrayObject,
  BooleanObject, 
  IndirectObject
)

# x1, y1, x2, y2 is bottom left corner and top right corner
def create_annotation(x, y, meta):
  color = [255.0 / 255.0, 209 / 255.0, 0]
  newAnnotation = DictionaryObject()
  # https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf
  newAnnotation.update({
    # Table 165 NoZoom
    NameObject("/F"): NumberObject(4),
    NameObject("/Type"): NameObject("/Annot"),
    NameObject("/Subtype"): NameObject("/Link"),
    # NameObject("/Subtype"): NameObject("/Text"),

    # Table 170 titlebar
    # NameObject("/T"): TextStringObject(meta["author"]),
    # NameObject("/Contents"): TextStringObject(meta["contents"]),
    
    # Table 164 color, annotation rectangle
    NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
    NameObject("/Rect"): ArrayObject([
      FloatObject(x),
      FloatObject(y-20),
      FloatObject(x+20),
      FloatObject(y)
    ]),

    # Table 173 link annotation
    NameObject('/A'): DictionaryObject({
      # Table 206 uri 
      NameObject('/S'): NameObject('/URI'),
      NameObject('/URI'): TextStringObject(meta["contents"])
    }),
    # Table 173 invert rect when mouse
    NameObject('/H'): NameObject('/I'),
    # table 164 hor corner radius, vert corner radius, border width
    # dash array table 56
    NameObject('/Border'): ArrayObject([
      NameObject(2), 
      NameObject(2), 
      NameObject(5), 
      ArrayObject([NameObject(3)])]),

    # 12.5.6.4 text annotation
    # NameObject('/Open'): BooleanObject(False),
    # NameObject('/Name'): NameObject('/Comment'),
  })

  return newAnnotation

def add_annotation_to_page(annotation, page, output):
  annotation_ref = output._addObject(annotation);

  if "/Annots" in page:
    page[NameObject("/Annots")].append(annotation_ref)
  else:
    page[NameObject("/Annots")] = ArrayObject([annotation_ref])