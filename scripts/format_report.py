"""Apply A4 pages, body font and page numbers without changing OOXML namespaces."""
from pathlib import Path
import zipfile,re
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'reports/REPORT.docx'
with zipfile.ZipFile(path) as z:data={name:z.read(name) for name in z.namelist()}
styles=data['word/styles.xml'].decode()
styles=re.sub(r'<w:rFonts\b[^>]*/>','<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>',styles)
def body_font(match):
 content=match.group(2)
 if '<w:rPr>' in content:content=content.replace('</w:rPr>','<w:sz w:val="24"/></w:rPr>')
 else:content+='<w:rPr><w:sz w:val="24"/></w:rPr>'
 return match.group(1)+content+match.group(3)
styles=re.sub(r'(<w:style\b[^>]*w:styleId="(?:Normal|BodyText|FirstParagraph)"[^>]*>)(.*?)(</w:style>)',body_font,styles,flags=re.S)
data['word/styles.xml']=styles.encode()
document=data['word/document.xml'].decode()
document=re.sub(r'<w:footerReference\b[^>]*r:id="rIdMaternalFooter"[^>]*/>','',document)
document=document.replace('<w:sectPr>','<w:sectPr><w:footerReference w:type="default" r:id="rIdMaternalFooter"/>')
document=re.sub(r'<w:pgSz\b[^>]*/>','',document);document=re.sub(r'<w:pgMar\b[^>]*/>','',document)
document=document.replace('</w:sectPr>','<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:bottom="1440" w:left="1440" w:right="1440" w:header="720" w:footer="720"/></w:sectPr>')
data['word/document.xml']=document.encode()
rels=data['word/_rels/document.xml.rels'].decode()
relationship='<Relationship Id="rIdMaternalFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer-maternal.xml"/>'
if 'rIdMaternalFooter' not in rels:rels=rels.replace('</Relationships>',relationship+'</Relationships>')
data['word/_rels/document.xml.rels']=rels.encode()
types=data['[Content_Types].xml'].decode()
if 'footer-maternal.xml' not in types:types=types.replace('</Types>','<Override PartName="/word/footer-maternal.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>');data['[Content_Types].xml']=types.encode()
data['word/footer-maternal.xml']=b'''<?xml version="1.0" encoding="UTF-8"?><w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:fldSimple w:instr="PAGE"><w:r><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>'''
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for name,content in data.items():z.writestr(name,content)
print('A4 pages and page numbers applied to report.')
