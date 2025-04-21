FROM public.ecr.aws/lambda/python:3.12

# Upgrade pip and wheel
# Install tar and findutils (for find command)
RUN microdnf install -y tar findutils && \
    microdnf clean all 
    
COPY requirements.txt .

# Step 1: Install everything except numpy (skip numpy here)
RUN pip install --no-cache-dir --upgrade -r requirements.txt -t /var/lang/lib/python3.12/site-packages

# Step 2: Force remove numpy that might have been installed by dependencies
RUN rm -rf /var/lang/lib/python3.12/site-packages/numpy* && \
    rm -rf /var/lang/lib/python3.12/site-packages/__pycache__/numpy*

# Step 3: Copy your custom pre-built numpy wheel
COPY numpy-2.2.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .

# Step 4: Install your custom wheel version of numpy
RUN pip install --no-cache-dir numpy-2.2.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -t /var/lang/lib/python3.12/site-packages
RUN find /var/lang/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + && \
    find /var/lang/lib/python3.12/site-packages -type f -name "*.pyc" -delete && \
    find /var/lang/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + && \
    find /var/lang/lib/python3.12/site-packages -type d -name "test" -exec rm -rf {} + && \
    find /var/lang/lib/python3.12/site-packages -type d -name "examples" -exec rm -rf {} + 
# Copy your actual Lambda code
COPY src/ .

# Step 5: Create tarball for Lambda layer
RUN mkdir -p /layer/python && \
cp -r /var/lang/lib/python3.12/site-packages/* /layer/python && \
cd /layer && tar -czf /python.tar.gz python

# Set the Lambda handler
CMD ["faq_handler.lambda_handler"]
