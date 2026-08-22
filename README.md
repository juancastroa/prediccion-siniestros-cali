# Pronóstico diario de siniestros viales — Cali

Dashboard que muestra cuántos siniestros viales se esperan mañana, calculado
con el modelo ARIMA(4,1,1) del proyecto, y que se actualiza solo todos los días.

## Cómo queda organizado

```
data/historico/    dataset base (2016 en adelante)
data/nuevos/        aquí se sube el archivo del día (CSV o Excel)
scripts/generar_pronostico.py   recalcula el pronóstico
docs/index.html      el dashboard (lo que ve el usuario final)
docs/prediccion.json  los datos que lee el dashboard (se genera solo)
.github/workflows/    la tarea programada que corre el script cada día
```

## Puesta en marcha (una sola vez)

1. Crea un repositorio en GitHub y sube esta carpeta completa.
2. Ve a **Settings → Pages**, y en "Build and deployment" selecciona
   la rama `main` y la carpeta `/docs`. Guarda. GitHub te dará una URL
   pública (algo como `https://tuusuario.github.io/tu-repo/`) — esa es
   la que le compartes al usuario final.
3. Ve a **Settings → Actions → General** y confirma que los workflows
   tienen permiso de "Read and write" (para que puedan guardar el
   pronóstico nuevo cada día).

Con eso ya queda funcionando: todos los días a las 5:00 a.m. (hora
Colombia) se recalcula el pronóstico solo, aunque no subas nada nuevo
ese día (usa el último dato disponible).

## Uso diario: subir los datos del día

1. Guarda el archivo del día (CSV o Excel) con las mismas columnas del
   dataset original — como mínimo necesita la columna `fecha`, una fila
   por siniestro.
2. Súbelo a la carpeta `data/nuevos/` del repositorio (por la web de
   GitHub, con "Add file → Upload files", o por Git si prefieres).
3. Al subirlo se dispara automáticamente la actualización del
   pronóstico (no hay que esperar a las 5 a.m.). El archivo se mueve
   luego a `data/historico/procesados/` para no volver a contarlo.

## Probarlo en tu computador antes de subirlo

```bash
pip install pandas numpy statsmodels scikit-learn openpyxl
python scripts/generar_pronostico.py
cd docs && python -m http.server 8000
```

Abre `http://localhost:8000` en el navegador.

## Limitaciones a tener en cuenta

- El modelo recalibra su rango de error usando los últimos 90 días de
  datos reales; si por algún motivo pasan varios días sin que se suba
  información nueva, el rango dejará de reflejar la situación actual.
- El pronóstico es de **un día hacia adelante** — no está pensado para
  proyectar semanas o meses.
- Es una estimación estadística, no una alerta de seguridad: el rango
  de 80% significa que en 8 de cada 10 días el número real cae dentro
  de ese rango, pero puede haber excepciones.
