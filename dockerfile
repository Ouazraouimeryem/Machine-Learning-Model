# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest



# Copy only the necessary files into the container at /app
COPY preprocess.py /app/
COPY train_model.py /app/
COPY predict_response.py /app/
COPY requirements.txt /app/



# Install any needed packages specified in requirements.txt
 RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary data file (intents.json) into the container at /app
COPY intents.json /app/

# Run data preprocessing
RUN python /app/preprocess.py

# Train the model
RUN python /app/train_model.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Run predict_response.py when the container launches
CMD ["python", "/app/predict_response.py"]
