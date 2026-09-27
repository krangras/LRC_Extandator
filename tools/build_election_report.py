from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader


VAULT = Path(r"C:\From d drive\Files and media\Obsidian Vault")
OUTPUT = VAULT / "Выборы 2026 Кирово-Чепецк — аналитический отчёт.pdf"

FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
pdfmetrics.registerFont(TTFont("Arial", FONT))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_BOLD))

PAGE_W, PAGE_H = A4
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#5D6D7E")
BLUE = colors.HexColor("#1769AA")
BLUE_LIGHT = colors.HexColor("#EAF3FA")
ORANGE = colors.HexColor("#E67E22")
LINE = colors.HexColor("#D5DDE5")
PAPER = colors.HexColor("#F7F9FB")


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(18 * mm, 17 * mm, PAGE_W - 36 * mm, PAGE_H - 34 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                      id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=draw_page)])


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - 5 * mm, PAGE_W, 5 * mm, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setFont("Arial", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, "Кирово-Чепецк • выборы 2026 • фактологический отчёт")
        canvas.drawRightString(PAGE_W - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", fontName="Arial-Bold", fontSize=29,
                          leading=33, textColor=INK, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle(name="CoverSub", fontName="Arial", fontSize=13,
                          leading=18, textColor=MUTED, spaceAfter=18))
styles.add(ParagraphStyle(name="H1x", fontName="Arial-Bold", fontSize=19,
                          leading=23, textColor=INK, spaceBefore=2, spaceAfter=10))
styles.add(ParagraphStyle(name="H2x", fontName="Arial-Bold", fontSize=12,
                          leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="Bodyx", fontName="Arial", fontSize=9.3,
                          leading=13.3, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", fontName="Arial", fontSize=7.8,
                          leading=10.5, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="CardTitle", fontName="Arial-Bold", fontSize=13,
                          leading=16, textColor=INK, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle(name="Metric", fontName="Arial-Bold", fontSize=18,
                          leading=20, textColor=BLUE, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName="Arial", fontSize=7.7,
                          leading=9, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Headerx", fontName="Arial-Bold", fontSize=8.2,
                          leading=10, textColor=colors.white))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def box(items, bg=BLUE_LIGHT, border=LINE, padding=9):
    t = Table([[items]], colWidths=[PAGE_W - 36 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, border),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]))
    return t


def data_table(headers, rows, widths=None):
    data = [[P(str(h), "Headerx") for h in headers]]
    for row in rows:
        data.append([P(str(x), "Bodyx") for x in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
    ]))
    return t


def fit_image(path, max_w, max_h):
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


cards = [
    (504, "проспект Кирова, д. 8 — Центр детского творчества «Радуга»", "Pasted image 20260921201351.png"),
    (505, "ул. Вятская Набережная, д. 5 — школа № 5, спортивный зал", "Pasted image 20260921201423.png"),
    (506, "ул. Вятская Набережная, д. 5 — школа № 5, малый спортивный зал", "Pasted image 20260921201517.png"),
    (507, "ул. Революции, д. 20 — Детская художественная школа им. Л. Т. Брылина", "Pasted image 20260921201542.png"),
    (509, "проспект Кирова, д. 1 — школа № 4", "Pasted image 20260921201610.png"),
    (510, "ул. Островского, д. 6 — ассоциация «Дружба»", "Pasted image 20260921201757.png"),
    (511, "ул. Терещенко, д. 13 — Центр образования им. Алексея Некрасова", "Pasted image 20260921201809.png"),
    (512, "ул. Азина, д. 1 — Детская школа искусств", "Pasted image 20260921201821.png"),
    (513, "проспект Мира, д. 37 — школа для обучающихся с ОВЗ", "Pasted image 20260921201833.png"),
    (514, "проезд Лермонтова, д. 3а — многофункциональный ресурсный центр", "Pasted image 20260921201843.png"),
    (515, "проезд Лермонтова, д. 1 — Центр образования им. Алексея Некрасова", "Pasted image 20260921201853.png"),
    (516, "проезд Лермонтова, д. 1 — Центр образования им. Алексея Некрасова", "Pasted image 20260921201902.png"),
    (517, "проспект Кирова, д. 27 — школа № 7", "Pasted image 20260921201910.png"),
    (518, "ул. Первомайская, д. 13 — Спортивная школа № 1", "Pasted image 20260921201920.png"),
    (519, "проспект Мира, д. 52 — гимназия № 1, большой спортзал", "Pasted image 20260921201934.png"),
]


def build():
    story = []
    story += [Spacer(1, 24 * mm), P("Выборы 2026", "CoverTitle"),
              P("Кирово-Чепецк: анализ 15 участков", "CoverSub"),
              HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=16),
              P("Краткий фактологический отчёт по присланным карточкам УИК №504–507 и №509–519. Материал описывает локальный городской срез и не заменяет официальную сводку по округу №107.", "Bodyx"),
              Spacer(1, 8 * mm)]
    metrics = [[P("15", "Metric"), P("21 756", "Metric"), P("9 585", "Metric"), P("44,06%", "Metric")],
               [P("участков", "MetricLabel"), P("зарегистрировано", "MetricLabel"), P("проголосовало", "MetricLabel"), P("явка", "MetricLabel")]]
    mt = Table(metrics, colWidths=[44 * mm] * 4)
    mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE_LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                            ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    story += [mt, Spacer(1, 9 * mm), box([P("Главное в двух строках", "H2x"), P("Срез неоднороден: на большинстве участков разрыв между лидерами умеренный, а УИК №507 заметно выделяется высокой явкой и результатом Андрея Березина. УИК №508 в исходных материалах отсутствует.", "Bodyx")]),
              Spacer(1, 8 * mm), P("Подготовлено по данным исходной заметки и прикреплённых изображений. Дата материалов: сентябрь 2026.", "Smallx"), PageBreak()]

    story += [P("Итоги", "H1x"), P("Сводные показатели 15 участков", "H2x"),
              data_table(["Показатель", "Голосов", "Доля"], [
                  ("Зарегистрировано избирателей", "21 756", "—"),
                  ("Проголосовало", "9 585", "44,06% явка"),
                  ("Андрей Березин", "2 602", "27,15%"),
                  ("Михаил Кремлёв", "2 110", "22,01%"),
                  ("Анна Альминова", "1 672", "17,44%"),
                  ("Остальные кандидаты вместе", "2 707", "28,24%"),
                  ("Недействительные", "494", "5,15%"),
              ], widths=[92 * mm, 32 * mm, 48 * mm]), Spacer(1, 7 * mm),
              box([P("Разрыв лидеров", "H2x"), P("Березин опережает Кремлёва на 492 голоса, или 5,13 процентного пункта. Если считать только действительные бюллетени (9 091), доли составляют: Березин — 28,62%, Кремлёв — 23,21%, Альминова — 18,39%, остальные — 29,78%.", "Bodyx")]),
              Spacer(1, 5 * mm),
              box([P("Контрфактический сценарий: 494 недействительных бюллетеня против ЕР", "H2x"),
                   P("Если условно считать, что все 494 недействительных бюллетеня были бы отданы Михаилу Кремлёву как единому кандидату «против ЕР», у него было бы 2 604 голоса против 2 602 у Березина. Разница составила бы 2 голоса: Кремлёв — 27,16%, Березин — 27,15% от всех 9 585 проголосовавших.", "Bodyx"),
                   P("Если не приписывать эти бюллетени одному кандидату, а считать их просто голосами не за ЕР, то доля всех остальных составила бы 6 983 голоса, или 72,85%. Это расчётный сценарий, а не установленный факт о намерениях избирателей.", "Bodyx")]),
              Spacer(1, 8 * mm), P("Наиболее заметные участки", "H2x"),
              data_table(["Показатель", "УИК", "Значение"], [
                  ("Максимальная явка", "507", "51,9%"),
                  ("Минимальная явка", "516", "38,7%"),
                  ("Максимальная доля Березина", "507", "38,6%"),
                  ("Минимальная доля Березина", "505", "23,5%"),
                  ("Максимальная доля Кремлёва", "514", "27,2%"),
                  ("Максимальная доля Альминовой", "517", "20,8%"),
              ], widths=[92 * mm, 25 * mm, 55 * mm]), PageBreak()]

    story += [P("Что видно по географии", "H1x"),
              P("Школа №5: УИК №505 и №506", "H2x"),
              P("Два участка находятся по одному адресу. В сумме Березин получил 279 голосов, Кремлёв — 277: разница всего 2 голоса при 1 155 проголосовавших. Альминова получила 213 голосов; совокупная явка — 44,25%.", "Bodyx"),
              P("Кластер Лермонтова: УИК №514–516", "H2x"),
              P("В сумме эти три участка дают 443 голоса за Березина против 431 за Кремлёва, то есть разрыв 12 голосов при 1 747 проголосовавших. Явка в кластере — около 40,2%.", "Bodyx"),
              P("УИК №507 как локальный выброс", "H2x"),
              P("Здесь одновременно самая высокая явка (51,9%) и самая высокая доля Березина (38,6%). Кремлёв получил 17,5%, Альминова — 14,8%; разрыв Березин–Кремлёв составил 156 голосов. Именно этот участок заметно влияет на общую картину.", "Bodyx"),
              P("Другие наблюдения", "H2x"),
              P("На 13 из 15 участков Березин имеет больше голосов, чем Кремлёв. Исключения: №506 (117 против 119) и №514 (162 против 167). На №517 Альминова получила 155 голосов (20,8%) и единственный раз превысила результат Кремлёва (131; 17,6%).", "Bodyx"),
              Spacer(1, 5 * mm), box([P("Как читать эти данные", "H2x"), P("Это описание наблюдаемых различий между участками, а не доказательство причин этих различий. Небольшой набор УИК нельзя автоматически обобщать на весь город или весь округ.", "Bodyx")]), PageBreak()]

    story += [P("Сравнение и ограничения", "H1x"),
              P("Городской срез и округ №107", "H2x"),
              data_table(["Кандидат", "15 УИК", "Округ №107*", "Разница"], [
                  ("Березин", "27,15%", "33,68%", "−6,53 п.п."),
                  ("Альминова", "17,44%", "18,64%", "−1,20 п.п."),
              ], widths=[62 * mm, 35 * mm, 42 * mm, 33 * mm]),
              P("* По опубликованной сводке после обработки 100% протоколов. Для Кремлёва в исходной заметке приведено только промежуточное сравнение: 12,04% по округу при обработке 70,34% протоколов против 22,01% в городском наборе; это не финальное сопоставление.", "Smallx"),
              P("Явка и результат", "H2x"),
              P("В исходном анализе корреляция явки с долей Березина оценена примерно как +0,50, но после исключения УИК №507 — около −0,03. Это показывает, насколько один необычный участок может менять статистическую картину. Для Кремлёва оценка связи с явкой отрицательная, однако 15 участков недостаточно для надёжных выводов о поведении избирателей.", "Bodyx"),
              P("Недействительные бюллетени", "H2x"),
              P("В среднем — 5,15% (494 бюллетеня), разброс — примерно от 3,1% до 6,6–6,7%. По этим данным нельзя делать выводы о нарушениях; для проверки нужны полные протоколы и дополнительные показатели.", "Bodyx"),
              P("Границы набора", "H2x"),
              P("В выборке 15 уникальных УИК: №504–507 и №509–519. УИК №508 результата не имеет. Округ №107 существенно шире Кирово-Чепецка, поэтому городской набор — локальный срез, а не результат всего округа.", "Bodyx"), PageBreak()]

    story += [P("Исходные карточки УИК", "H1x"), P("Все 15 изображений, использованных в заметке", "Smallx"), Spacer(1, 3 * mm)]
    for idx, (num, address, filename) in enumerate(cards):
        path = VAULT / filename
        if not path.exists():
            continue
        story += [KeepTogether([P(f"УИК №{num}", "CardTitle"), P(address, "Smallx"), fit_image(path, 145 * mm, 190 * mm)])]
        if idx != len(cards) - 1:
            story.append(PageBreak())

    doc = ReportDocTemplate(str(OUTPUT), pagesize=A4, title="Выборы 2026 — Кирово-Чепецк")
    doc.build(story)
    print(str(OUTPUT).encode("ascii", "backslashreplace").decode("ascii"))


if __name__ == "__main__":
    build()
