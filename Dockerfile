# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the commits were labeled in: its Python and SQLite
# decide which sqlglot tests run, so keep it since the selector runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The labeling environment: these packages plus the sqlglot checkout on PYTHONPATH.
COPY requirements.lock /build/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /build/requirements.lock

# The base image ships the standard library without bytecode and the run's disk is read-only:
# every process would compile asyncio, unittest.mock, multiprocessing... from source each time
# it imports them (0.09 s for unittest.mock alone). Files that are not valid Python are skipped.
RUN python -m compileall -q -x 'site-packages' /usr/local/lib/python3.13 > /dev/null 2>&1 || true

# sqlglot at the pinned commit, from its own GitHub repository, with bytecode
# compiled now because the disk is read-only at run time.
COPY tools/fetch_sqlglot.py /build/fetch_sqlglot.py
RUN python /build/fetch_sqlglot.py /opt/sqlglot \
    && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests \
    && chmod -R a+rX /opt/sqlglot

ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0

COPY app /app
# Fingerprints of the pinned checkout's import-time state (see app/fingerprint.py), and the
# map in the form that starts fastest and the index of the checkout's identifiers (see
# app/packmap.py; without them the selector reads map.pkl.gz and builds the index itself),
# and what the executor tests' set-up makes of the pinned TPC fixtures (app/tpcmemo.py; without
# it their check runs parse the csv files and ask duckdb themselves).
RUN PYTHONPATH=/opt/sqlglot python /app/fingerprint.py /app/state.pkl \
    && (python /app/packmap.py /app/map.pkl.gz /app/map.bin || rm -f /app/map.bin) \
    && (python /app/packmap.py --words /opt/sqlglot /app/words.bin || rm -f /app/words.bin) \
    && (cd /opt/sqlglot && PYTHONPATH=/opt/sqlglot python /app/tpcmemo.py /app/tpcmemo || rm -rf /app/tpcmemo) \
    && python -m compileall -q /app && chmod -R a+rX /app
CMD ["python", "/app/wrap.py"]
