"""
Precarga inicial (seed) idempotente:
  - Agrega OBLIGATORIAMENTE doc.alvaro.pinto.ma@unifranz.edu.bo como administrador.
  - Agrega correos extra configurados en EXTRA_ADMIN_EMAILS y EXTRA_OPERATOR_EMAILS.
  - Precarga casetas, tabla de tarifas por categoría/ejes y 25 vehículos de prueba (>= 20).
"""

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Caseta, Tarifa, Usuario, Vehiculo

DOCENTE_EMAIL = "doc.alvaro.pinto.ma@unifranz.edu.bo"

DEFAULT_CASETAS = [
    ("Caseta Autopista Norte", "Autopista La Paz - El Alto, Km 5"),
    ("Caseta Carretera Sur", "Carretera La Paz - Oruro, Peaje Senkata"),
    ("Caseta Puente Sacaba", "Av. Villazón Km 8, Cochabamba"),
]

DEFAULT_TARIFAS = [
    ("moto", 1, 1.5),
    ("moto", 2, 2.0),
    ("auto", 1, 3.0),
    ("auto", 2, 5.0),
    ("minibus", 2, 8.0),
    ("bus", 2, 15.0),
    ("bus", 3, 22.0),
    ("camion", 1, 12.0),
    ("camion", 2, 18.0),
    ("camion", 3, 28.0),
    ("camion", 4, 38.0),
    ("camion", 5, 48.0),
    ("camion", 6, 60.0),
]

# 25 vehículos de prueba (supera el mínimo obligatorio de 20 vehículos)
DEFAULT_VEHICULOS = [
    ("2481KLM", "Carlos Mendoza Quispe", "auto", 2, 150.0, True),
    ("3092XTR", "María Elena Vargas", "auto", 2, 85.5, True),
    ("4510BGD", "Transportes Andino S.R.L.", "camion", 3, 420.0, True),
    ("5821LPZ", "Cooperativa 16 de Julio", "minibus", 2, 64.0, True),
    ("6190CBX", "Flota Bolívar S.A.", "bus", 3, 500.0, True),
    ("1204MTO", "Diego Rojas Flores", "moto", 2, 30.0, True),
    ("7734SRZ", "Logística del Oriente", "camion", 4, 600.0, True),
    ("8820TRJ", "Ana Lucía Paredes", "auto", 2, 0.0, True),  # Sin saldo (prueba alerta)
    ("9015PTS", "Roberto Mamani Condori", "minibus", 2, 3.0, True),  # Saldo insuficiente (prueba alerta)
    ("1842ORU", "Trans Copacabana MEM", "bus", 2, 210.0, True),
    ("2938BNE", "Sofía Castro Salazar", "auto", 2, 95.0, True),
    ("3317PDN", "José Antonio Gutiérrez", "moto", 2, 20.0, True),
    ("4091CHQ", "Constructora Altiplano", "camion", 3, 350.0, True),
    ("5129LPN", "Sindicato Eduardo Avaroa", "minibus", 2, 48.0, True),
    ("6408SCZ", "Camila Morales Torrez", "auto", 2, 120.0, True),
    ("7250CBB", "Transporte Pesado Illimani", "camion", 5, 780.0, True),
    ("8193LPA", "Fernando Chávez Lima", "auto", 2, 45.0, True),
    ("9420BTA", "Expreso del Sur S.R.L.", "bus", 3, 310.0, True),
    ("1576KRS", "Valeria Huanca Nina", "moto", 2, 15.0, True),
    ("2684DPL", "Jorge Luis Peñaranda", "auto", 2, 200.0, True),
    ("3749FGH", "Distribuidora Nacional", "camion", 2, 180.0, True),
    ("4852JKL", "Sindicato Mixto Murillo", "minibus", 2, 72.0, True),
    ("5963QWE", "Patricia Zambrana", "auto", 2, 0.0, True),  # Sin saldo
    ("6074RTY", "Carga Pesada Bolivia", "camion", 6, 900.0, True),
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
        if not db.query(Vehiculo).filter(Vehiculo.placa == placa).first():
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

    db.commit()
