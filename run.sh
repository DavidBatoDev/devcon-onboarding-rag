#!/bin/bash
# Started by Lambda Web Adapter (AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap, handler=run.sh).
PATH=$PATH:$LAMBDA_TASK_ROOT/bin \
    PYTHONPATH=$PYTHONPATH:/opt/python:$LAMBDA_RUNTIME_DIR \
    exec python -m uvicorn --host 127.0.0.1 --port=$PORT --no-access-log main:app
