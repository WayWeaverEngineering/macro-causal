# Use the official EMR Serverless Spark base image
FROM public.ecr.aws/emr-serverless/spark/emr-6.15.0:latest

# Set working directory
WORKDIR /opt/spark/work-dir

# Install additional Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt || python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py /opt/spark/work-dir/

# Set environment variables
ENV PYTHONPATH=/opt/spark/work-dir
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Make main.py executable
RUN chmod +x /opt/spark/work-dir/main.py

# Default command (will be overridden by spark-submit)
CMD ["python", "/opt/spark/work-dir/main.py"]