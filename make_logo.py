from PIL import Image, ImageDraw, ImageFont

# Création d'une bannière logo professionnelle (800 x 180 pixels)
W, H = 800, 180
img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# 1. Carré arrondi Bleu Nuit (#0F2C59) à gauche
draw.rounded_rectangle([10, 20, 150, 160], radius=28, fill="#0F2C59", outline="#F59E0B", width=4)

# 2. Bouclier doré et coche verte à l'intérieur
shield = [(80, 42), (120, 58), (120, 92), (80, 138), (40, 92), (40, 58)]
draw.polygon(shield, fill="#1E4B8F", outline="#F59E0B")
draw.line([(60, 90), (75, 106), (104, 72)], fill="#10B981", width=9)

# 3. Chargement des polices
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
except:
    font_title = ImageFont.load_default()
    font_badge = ImageFont.load_default()
    font_sub = ImageFont.load_default()

# 4. Écriture du nom "TerangaScore" + Badge "PRO" + Slogan
draw.text((175, 32), "Teranga", fill="#0F2C59", font=font_title)
draw.text((405, 32), "Score", fill="#D97706", font=font_title)

draw.rounded_rectangle([578, 42, 658, 82], radius=12, fill="#10B981")
draw.text((592, 48), "PRO", fill="#FFFFFF", font=font_badge)

draw.text((177, 105), "Scoring Prédictif & Inclusion Financière • Dakar", fill="#4B5563", font=font_sub)

img.save("logo_terangascore.png")
print("Logo PNG généré avec succès !")
