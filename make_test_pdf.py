"""Synthetic IMF-style Article IV PDF for offline smoke tests (structure mirrors Country Report 24/192).
It reproduces the real document's traps: running headers, unicode fractions, TOC leader lines, part covers that
repeat 'STAFF REPORT FOR THE 2024 ARTICLE IV', and a page where Table 5b's block is written to the content
stream BEFORE Table 5a's block although it sits lower on the page."""
import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

W, H = letter
FONT, SIZE = "Helvetica", 10
P = ("Dominica's output has recovered to its pre-pandemic level, reflecting a rebound in tourism and public investment "
     "supported by buoyant Citizenship by Investment (CBI) revenues. Inflation fell from its 2022 peak of 9¾ percent to "
     "2¼ percent by end-2023. Banks remain well capitalized and liquid although credit unions suffer from weak capital "
     "and asset quality. Public debt rose by 43½ percentage points at the pandemic peak and remains high at 97.5 percent "
     "of GDP. The outlook is subject to downside risks from climate change, volatile tourism receipts and CBI revenues. "
     "Staff recommend stepping up provisioning, strengthening supervision of credit unions and rebuilding fiscal buffers "
     "to reach the ECCU debt target of 60 percent of GDP by 2035. Youth unemployment is high and labor force "
     "participation is low, while housing construction and mortgage lending remain subdued.")


class Doc:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=letter)
        self.page = 0
        self.y = H - 72
        self.header = False

    def new_page(self, header=True):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.y = H - 60
        self.c.setFont(FONT, SIZE)
        if header:
            self.c.drawString(72, H - 40, "DOMINICA")
            pn = self.page
            self.c.drawRightString(W - 72, H - 40, f"INTERNATIONAL MONETARY FUND {pn}" if pn % 2 else f"{pn} INTERNATIONAL MONETARY FUND")
            self.y = H - 75

    def line(self, text, bold=False, x=72, gap=13):
        self.c.setFont("Helvetica-Bold" if bold else FONT, SIZE)
        self.c.drawString(x, self.y, text)
        self.y -= gap

    def para(self, text):
        for l in simpleSplit(text, FONT, SIZE, W - 144):
            if self.y < 72:
                self.new_page()
            self.line(l)
        self.y -= 6

    def table(self, title, years, rows, x0=72):
        self.line(title, bold=True)
        self.line("".join(f"{y:>8}" for y in years), x=230)
        for label, vals in rows:
            if vals is None:
                self.line(label)
                continue
            self.c.drawString(x0, self.y, label)
            for k, v in enumerate(vals):
                self.c.drawRightString(260 + 30 * k, self.y, v)
            self.y -= 12


def build(path):
    d = Doc(path)
    d.new_page(header=False)
    for t in ["© 2024 International Monetary Fund", "IMF Country Report No. 24/192", "DOMINICA",
              "2024 ARTICLE IV CONSULTATION-PRESS RELEASE; STAFF REPORT; AND STATEMENT BY THE EXECUTIVE",
              "DIRECTOR FOR DOMINICA", "June 2024"]:
        d.line(t)
    d.new_page(header=False)
    d.line("PR 24/247")
    d.line("IMF Executive Board Concludes 2024 Article IV Consultation with Dominica")
    d.line("FOR IMMEDIATE RELEASE")
    for _ in range(3):
        d.para(P)
    yrs = list(range(2019, 2030))
    t1 = [("Output and prices (annual percent change, unless otherwise specified)", None),
          ("Real GDP 1/", "5.5 -16.6 6.9 5.6 4.7 4.6 4.3 3.3 2.9 2.9 2.4".split()),
          ("Consumer prices", None),
          ("Period average", "1.5 -0.7 1.6 7.7 3.5 2.8 2.1 2.0 2.0 2.0 2.0".split()),
          ("End of period", "0.1 -0.7 3.5 8.7 2.3 2.2 2.0 2.0 2.0 2.0 2.0".split()),
          ("Central government debt (incl. guaranteed) 3/", "97.7 112.5 108.5 103.3 97.5 98.8 95.9 93.8 91.7 89.6 87.4".split()),
          ("Money and credit (annual percent change)", None),
          ("Credit to the private sector", "-6.0 -0.3 3.6 2.7 -3.6 4.0 5.3 5.4 5.0 7.1 6.5".split()),
          ("Nominal GDP (EC$ millions)", "1,651 1,361 1,499 1,640 1,777 1,911 2034 2142 2249 2361 2466".split())]
    d.new_page(header=False)
    d.line("Table 1. Dominica: Selected Economic Indicators, 2019–29", bold=True)
    d.line("Est. Projected", x=380)
    d.line(" ".join(map(str, yrs)), x=230)
    for label, vals in t1:
        if vals is None:
            d.line(label)
        else:
            d.line(f"{label} {' '.join(vals)}")
    d.line("1/ At market prices.")
    # Staff report cover
    d.new_page(header=False)
    d.line("DOMINICA")
    d.line("STAFF REPORT FOR THE 2024 ARTICLE IV CONSULTATION")
    d.line("KEY ISSUES")
    d.para("Context. " + P)
    d.para("Outlook and Risks. " + P)
    d.new_page()
    d.line("CONTENTS")
    for t, pg in [("CONTEXT", 4), ("RECENT DEVELOPMENTS", 4), ("OUTLOOK AND RISKS", 6), ("POLICY DISCUSSIONS", 8),
                  ("STAFF APPRAISAL", 19)]:
        d.line(f"{t} {'_' * 60} {pg}")
    for h in ["CONTEXT", "RECENT DEVELOPMENTS", "OUTLOOK AND RISKS", "POLICY DISCUSSIONS"]:
        d.new_page()
        d.line(h, bold=True)
        if h == "POLICY DISCUSSIONS":
            d.line("A. Safeguarding Fiscal Sustainability and Rebuilding Buffers", bold=True)
        for k in range(1, 5):
            d.para(f"{k}. " + P)
        if h == "POLICY DISCUSSIONS":
            d.line("B. Strengthening Financial System Balance Sheets and Intermediation", bold=True)
            for k in range(5, 9):
                d.para(f"{k}. " + P)
    d.new_page()
    d.line("STAFF APPRAISAL", bold=True)
    for k in range(20, 24):
        d.para(f"{k}. " + P)
    d.new_page()
    d.line("Figure 1. Dominica: Real Sector Developments", bold=True)
    d.line("Growth has recovered to pre-pandemic levels, supported by tourism and investment.")
    for frag in ["Real GDP", "(percent)", "2019", "2021", "2023", "Tourism", "Arrivals", "(thousands)", "CPI", "Fuel"]:
        d.line(frag, x=120)
    d.line("Inflation has subsided from its 2022 peak, reflecting lower commodity prices.")
    d.new_page()
    d.line("Table 1. Dominica: Selected Economic Indicators, 2019–29", bold=True)
    d.line("Area (sq. km.) 754 Adult literacy rate (percent, 2016) 94")
    d.line("Unemployment rate (2016) 23 Population (2022) 72,737")
    d.line(" ".join(map(str, yrs)), x=230)
    for label, vals in t1:
        if vals is None:
            d.line(label)
        else:
            d.line(f"{label} {' '.join(vals)}")
    # FSI page with stream-order trap: draw Table 5b's rows first, lower on the page
    d.new_page()
    c = d.c
    fsi_years = list(range(2013, 2024))
    c.setFont("Helvetica-Bold", SIZE)
    c.drawString(72, 700, "Table 5a. Dominica: Financial Stability Indicators-Commercial Banks, 2013–23")
    c.drawString(72, 470, "Table 5b. Dominica: Financial Stability Indicators-Credit Unions, 2013–23")
    c.setFont(FONT, SIZE)

    def block(ytop, rows):
        c.drawString(230, ytop, " ".join(map(str, fsi_years)))
        y = ytop - 14
        for label, vals in rows:
            c.drawString(72, y, label + ("" if vals is None else " " + " ".join(vals)))
            y -= 13
    block(450, [("Capital Adequacy", None), ("Regulatory capital ratio", "8.45 8.65 3.38 8.75 8.13 7.64 7.58 8.19 8.48 8.17 8.47".split()),
                ("Asset quality", None), ("Nonperforming loans to total gross loans", "7.59 8.78 40.93 14.53 15.06 13.89 12.70 15.31 16.79 14.61 13.41".split())])
    block(680, [("Capital Adequacy", None), ("Regulatory capital ratio 1/", "13.1 11.9 12.9 15.6 13.3 12.5 13.9 18.9 18.3 15.9 27.51".split()),
                ("Asset quality", None), ("Nonperforming loans to total gross loans", "13.9 14.2 15.1 14.5 17.4 17.0 12.2 15.0 15.7 13.8 12.9".split())])
    c.drawString(72, 520, "Source: ECCB")
    c.drawString(72, 508, "1/ Total Capital over risk weighted assets.")
    for k, ttl in enumerate(["Annex I. External Sector Assessment", "Annex II. Risk Assessment Matrix",
                             "Annex V. The Role of Financial Intermediation and Growth in Dominica"]):
        d.new_page()
        d.line(ttl, bold=True)
        for _ in range(3):
            d.para(P)
    d.new_page(header=False)
    d.line("DOMINICA")
    d.line("STAFF REPORT FOR THE 2024 ARTICLE IV CONSULTATION-")
    d.line("INFORMATIONAL ANNEX")
    d.line("FUND RELATIONS")
    d.para("Membership Status: Joined December 12, 1978. " + P)
    d.new_page(header=False)
    d.line("DOMINICA")
    d.line("STAFF REPORT FOR THE 2024 ARTICLE IV")
    d.line("CONSULTATION-DEBT SUSTAINABILITY ANALYSIS1")
    for _ in range(3):
        d.para("Dominica's debt is sustainable, but the country remains at high risk of debt distress. The stress tests "
               "include a natural disaster shock and a combined contingent liability shock. " + P)
    d.new_page(header=False)
    d.line("Statement by Mr. Philip Jennings and Ms. Sylvia Gumbs on Dominica")
    d.line("May 31, 2024")
    for _ in range(3):
        d.para("On behalf of our Dominican authorities, we thank staff. " + P)
    d.c.save()
    print(f"wrote {path} ({d.page} pages)")


if __name__ == "__main__":
    build(sys.argv[1])
