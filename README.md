# BrickSmart-Text to LEGO Tutorial Generator (Backend)

This project provides a Django-based backend for generating 3D models from text and images, converting these models into LEGO brick instructions, and providing guided assembly steps.

## Features

- **Text-to-3D Model**: Generate 3D models using OpenAI's GPT models based on textual prompts.
- **Image-to-3D Model**: Upload an image to generate 3D models using the Tripo3D API.
- **3D Model to LEGO**: Convert 3D models into voxel representations and generate LEGO building instructions.
- **Guided Assembly**: Use AI to provide guided assembly steps, especially for children.

## Installation

### Prerequisites

- Python 3.10
- Django 5.0.6
- Pipenv or virtualenv
- SQLite (for development)

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/3d-lego-tutorial-generator.git
    cd 3d-lego-tutorial-generator
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    pipenv install
    pipenv shell
    ```

    or using `virtualenv`:

    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

3. **Set up the database:**

    ```bash
    python manage.py migrate
    ```

4. **Run the development server:**

    ```bash
    python manage.py runserver
    ```

    The server should now be running at `http://127.0.0.1:8000/`.

### Configuration

- **Environment Variables**: Set up your environment variables in a `.env` file for API keys and other secrets.

    ```bash
    OPENAI_API_KEY=<your_openai_api_key>
    QINIU_ACCESS_KEY=<your_qiniu_access_key>
    QINIU_SECRET_KEY=<your_qiniu_secret_key>
    ```

- **API Keys**: Ensure you have the necessary API keys for OpenAI, Qiniu Cloud, and Tripo3D API.

### Usage

1. **Generate a 3D Model from Text:**

    - Navigate to the prompt page and enter a descriptive prompt.
    - The system will generate a 3D model based on the prompt.

2. **Generate a 3D Model from an Image:**

    - Upload an image file on the image upload page.
    - The system will generate a 3D model based on the uploaded image.

3. **Convert a 3D Model to LEGO:**

    - Use the generated 3D model and follow the steps to convert it to LEGO building instructions.
    - The system provides a step-by-step guide for assembling the LEGO model.

### Contributing

Feel free to fork this repository, make your changes, and submit a pull request.

### License

This project is licensed under the MIT License.

### Contact

For any questions or suggestions, please open an issue or contact the repository owner.
