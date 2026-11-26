"""
Precarga inicial (seed) idempotente:
  - Agrega OBLIGATORIAMENTE doc.alvaro.pinto.ma@unifranz.edu.bo como administrador.
  - Agrega correos extra configurados en EXTRA_ADMIN_EMAILS y EXTRA_OPERATOR_EMAILS.
  - Precarga casetas, tabla de tarifas por categoría/ejes y 25 vehículos de prueba (>= 20).
"""

import hashlib
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Caseta, Paso, Tarifa, Usuario, Vehiculo

DOCENTE_EMAIL = "doc.alvaro.pinto.ma@unifranz.edu.bo"

NOMBRES_FICTICIOS_PARTICULARES = [
    "Rodrigo Alejandro Mamani Quispe",
    "Valeria Fernanda Torrez Flores",
    "Mauricio Esteban Rojas Condori",
    "Daniela Alejandra Vargas Rocha",
    "Camila Lucía Morales Choque",
    "Javier Eduardo Peñaranda Lima",
    "Paola Andrea Gutiérrez Nina",
    "Álvaro Sebastián Aguilar Castro",
    "Gabriela Sofía Chávez Huanca",
    "Cristian David Salazar Ticona",
    "Lorena Natalia Zambrana Paco",
    "Jorge Andrés Limachi Torrico",
    "Miriam Beatriz Apaza Callisaya",
    "Fernando José Mendoza Paredes",
    "Natalia Carolina Quiroga Soto",
    "Diego Armando Velasco Guzmán",
]

CONDUCTORES_TRANSPORTE = [
    "Juan Carlos Quispe",
    "Rubén Darío Condori",
    "Marco Antonio Mamani",
    "Héctor Hugo Choque",
    "Ramiro Luis Gutiérrez",
    "Óscar Eduardo Peñaranda",
    "Víctor Hugo Ticona",
    "Miguel Ángel Limachi",
    "Freddy Reynaldo Callisaya",
    "Luis Alberto Huanca",
    "Édgar Iván Apaza",
    "César Augusto Paco",
    "Gustavo Adolfo Torrico",
    "Reynaldo Félix Nina",
]

SINDICATOS_MINIBUS = [
    "Sindicato Mixto Señor de Mayo",
    "Sindicato de Transporte Villa Victoria",
    "Sindicato Eduardo Avaroa",
    "Cooperativa de Transporte 14 de Septiembre",
    "Sindicato Mixto Litoral",
    "Sindicato 24 de Julio",
    "Sindicato Pedro Domingo Murillo",
    "Cooperativa Unión Altiplano",
]

SINDICATOS_CAMION = [
    "Sindicato de Transporte Pesado Illimani",
    "Sindicato Pesado Trans Altiplano Sur",
    "Federación de Camioneros 16 de Noviembre",
    "Sindicato de Carga Pesada Sajama",
    "Cooperativa Transporte Pesado Andino",
    "Sindicato Mixto Carga Real Bolivia",
    "Sindicato Pesado Estrella del Oriente",
    "Asociación Transporte Pesado Tunari",
]

SINDICATOS_BUS = [
    "Sindicato Flota Trans Copacabana",
    "Sindicato Mixto Expreso Bolívar",
    "Cooperativa Interdepartamental El Dorado",
    "Sindicato Flota Trans Illimani",
]

INITIAL_SALDO_BY_CATEGORY = {
    "moto": 80.0,
    "auto": 160.0,
    "minibus": 240.0,
    "bus": 450.0,
    "camion": 580.0,
}


def generate_fictional_owner(placa: str, categoria: str) -> str:
    """
    Genera de forma determinista un propietario ficticio realista según la clase del vehículo:
      - Auto / Moto: nombre completo de persona particular.
      - Minibús / Camión / Bus: nombre del socio conductor + nombre del Sindicato / Cooperativa.
    """
    digest = hashlib.sha256(f"{placa}:{categoria}".encode("utf-8")).digest()
    h1 = int.from_bytes(digest[:4], "big")
    h2 = int.from_bytes(digest[4:8], "big")
    cat = (categoria or "auto").lower().strip()

    if cat == "minibus":
        chofer = CONDUCTORES_TRANSPORTE[h1 % len(CONDUCTORES_TRANSPORTE)]
        sind = SINDICATOS_MINIBUS[h2 % len(SINDICATOS_MINIBUS)]
        return f"{chofer} — {sind}"
    if cat == "camion":
        chofer = CONDUCTORES_TRANSPORTE[h1 % len(CONDUCTORES_TRANSPORTE)]
        sind = SINDICATOS_CAMION[h2 % len(SINDICATOS_CAMION)]
        return f"{chofer} — {sind}"
    if cat == "bus":
        chofer = CONDUCTORES_TRANSPORTE[h1 % len(CONDUCTORES_TRANSPORTE)]
        sind = SINDICATOS_BUS[h2 % len(SINDICATOS_BUS)]
        return f"{chofer} — {sind}"
    return NOMBRES_FICTICIOS_PARTICULARES[h1 % len(NOMBRES_FICTICIOS_PARTICULARES)]

DEFAULT_CASETAS = [
    ("Caseta Autopista Norte", "Autopista La Paz - El Alto, Km 5"),
    ("Caseta Carretera Sur", "Carretera La Paz - Oruro, Peaje Senkata"),
    ("Caseta Puente Sacaba", "Av. Villazón Km 8, Cochabamba"),
]

DEFAULT_TARIFAS = [
    ("moto", 1, 2.0),
    ("moto", 2, 2.0),
    ("auto", 1, 5.0),
    ("auto", 2, 5.0),
    ("minibus", 1, 8.0),
    ("minibus", 2, 8.0),
    ("bus", 2, 15.0),
    ("bus", 3, 22.0),
    ("camion", 1, 15.0),
    ("camion", 2, 18.0),
    ("camion", 3, 28.0),
    ("camion", 4, 38.0),
    ("camion", 5, 48.0),
    ("camion", 6, 60.0),
]

# 25 vehículos de prueba (particulares, minibuses y motos con 1 eje; camiones y buses con 2+ ejes)
DEFAULT_VEHICULOS = [
    ("2481KLM", "Carlos Mendoza Quispe", "auto", 2, 150.0, True),
    ("3092XTR", "María Elena Vargas", "auto", 2, 85.5, True),
    ("4510BGD", "Ramiro Gutiérrez — Sindicato de Transporte Pesado Illimani", "camion", 3, 420.0, True),
    ("5821LPZ", "Juan Carlos Quispe — Cooperativa 14 de Septiembre", "minibus", 2, 64.0, True),
    ("6190CBX", "Fernando Salazar — Sindicato Expreso Bolívar", "bus", 3, 500.0, True),
    ("1204MTO", "Diego Rojas Flores", "moto", 2, 30.0, True),
    ("7734SRZ", "Óscar Peñaranda — Sindicato Pesado Estrella del Oriente", "camion", 4, 600.0, True),
    ("8820TRJ", "Ana Lucía Paredes", "auto", 2, 0.0, True),  # Sin saldo (prueba alerta)
    ("9015PTS", "Roberto Mamani — Sindicato Mixto Señor de Mayo", "minibus", 2, 3.0, True),  # Saldo insuficiente (prueba alerta)
    ("1842ORU", "Gustavo Torrico — Sindicato Flota Trans Copacabana", "bus", 2, 210.0, True),
    ("2938BNE", "Sofía Castro Salazar", "auto", 2, 95.0, True),
    ("3317PDN", "José Antonio Gutiérrez", "moto", 2, 20.0, True),
    ("4091CHQ", "Víctor Hugo Ticona — Sindicato Pesado Trans Altiplano Sur", "camion", 3, 350.0, True),
    ("5129LPN", "Marco Antonio Choque — Sindicato Eduardo Avaroa", "minibus", 2, 48.0, True),
    ("6408SCZ", "Camila Morales Torrez", "auto", 2, 120.0, True),
    ("7250CBB", "Miguel Ángel Limachi — Federación de Camioneros 16 de Noviembre", "camion", 5, 780.0, True),
    ("8193LPA", "Fernando Chávez Lima", "auto", 2, 45.0, True),
    ("9420BTA", "César Augusto Paco — Cooperativa Interdepartamental El Dorado", "bus", 3, 310.0, True),
    ("1576KRS", "Valeria Huanca Nina", "moto", 2, 15.0, True),
    ("2684DPL", "Jorge Luis Peñaranda", "auto", 2, 200.0, True),
    ("3749FGH", "Freddy Callisaya — Cooperativa Transporte Pesado Andino", "camion", 2, 180.0, True),
    ("4852JKL", "Rubén Darío Condori — Sindicato Pedro Domingo Murillo", "minibus", 2, 72.0, True),
    ("5963QWE", "Patricia Zambrana", "auto", 2, 0.0, True),  # Sin saldo
    ("6074RTY", "Reynaldo Nina — Sindicato de Carga Pesada Sajama", "camion", 6, 900.0, True),
    ("7185UIO", "Vehículo Dado de Baja", "auto", 2, 50.0, False),  # Inactivo
]


def _ensure_user(db: Session, email: str, nombre: str, rol: str, seen_emails: set):
    settings = get_settings()
    email_clean = email.strip().lower()
    if not email_clean or email_clean in seen_emails:
        return
    if settings.allowed_email_domain and not email_clean.endswith(f"@{settings.allowed_email_domain}"):
        return
    seen_emails.add(email_clean)
    existing = db.query(Usuario).filter(Usuario.email_google == email_clean).first()
    if not existing:
        db.add(
            Usuario(
                email_google=email_clean,
                nombre=nombre.strip(),
                rol=rol,
                activo=True,
            )
        )
        db.flush()


def seed_database(db: Session) -> None:
    settings = get_settings()
    seen_emails: set = set()

    # 1. Docente administrador obligatorio (Sección 4.4)
    _ensure_user(
        db,
        email=DOCENTE_EMAIL,
        nombre="Doc. Álvaro Pinto (Docente UNIFRANZ)",
        rol="administrador",
        seen_emails=seen_emails,
    )

    # Administradores adicionales desde .env (ignora duplicados del docente o repetidos)
    for raw_email in settings.extra_admin_emails.split(","):
        clean = raw_email.strip()
        if clean:
            _ensure_user(db, clean, clean.split("@")[0].title(), "administrador", seen_emails)

    # Operadores adicionales desde .env
    for raw_email in settings.extra_operator_emails.split(","):
        clean = raw_email.strip()
        if clean:
            _ensure_user(db, clean, clean.split("@")[0].title(), "operador", seen_emails)

    # 2. Casetas
    for nombre, ubicacion in DEFAULT_CASETAS:
        if not db.query(Caseta).filter(Caseta.nombre == nombre).first():
            db.add(Caseta(nombre=nombre, ubicacion=ubicacion))

    # 3. Tarifas
    for categoria, ejes, monto in DEFAULT_TARIFAS:
        exists = (
            db.query(Tarifa)
            .filter(Tarifa.categoria == categoria, Tarifa.ejes == ejes)
            .first()
        )
        if not exists:
            db.add(Tarifa(categoria=categoria, ejes=ejes, monto=monto))

    # 4. Vehículos (>= 20 vehículos de prueba)
    for placa, propietario, categoria, ejes, saldo, activo in DEFAULT_VEHICULOS:
        existing_v = db.query(Vehiculo).filter(Vehiculo.placa == placa).first()
        if not existing_v:
            db.add(
                Vehiculo(
                    placa=placa,
                    propietario=propietario,
                    categoria=categoria,
                    ejes=ejes,
                    saldo=saldo,
                    activo=activo,
                )
            )
        elif categoria in ("minibus", "camion", "bus") and "—" not in existing_v.propietario:
            existing_v.propietario = propietario

    # 5. Empadronar retroactivamente pasos previos en BD que aún no tenían vehículo registrado
    known_plates = {row[0] for row in db.query(Vehiculo.placa).all() if row[0]}
    prev_pasos = db.query(Paso).order_by(Paso.created_at.desc()).all()
    for p in prev_pasos:
        pl = (p.placa_normalizada or "").strip()
        if len(pl) >= 4 and pl not in known_plates:
            cat = (p.categoria_detectada or "auto").lower()
            ej = max(2, int(p.ejes_detectados or 2))
            saldo_ini = max(20.0, INITIAL_SALDO_BY_CATEGORY.get(cat, 150.0) - float(p.monto or 0.0))
            db.add(
                Vehiculo(
                    placa=pl,
                    propietario=generate_fictional_owner(pl, cat),
                    categoria=cat,
                    ejes=ej,
                    saldo=round(saldo_ini, 2),
                    activo=True,
                )
            )
            known_plates.add(pl)
            if p.estado == "NO_REGISTRADO":
                p.estado = "APROBADO"

    db.commit()
