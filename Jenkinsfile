pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out Airman project...'
                checkout scm
            }
        }

        stage('Check Python') {
            steps {
                echo 'Checking Python version...'
                bat 'python --version'
                bat 'python -m ensurepip --upgrade'
                bat 'python -m pip --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                bat 'python -m pip install --upgrade pip'
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('Check Project Structure') {
            steps {
                echo 'Checking Airman project files...'
                bat 'dir'
                bat 'dir src'
                bat 'dir src\\model'
            }
        }

        stage('Check Python Imports') {
            steps {
                echo 'Checking important Python imports...'
                bat 'python -c "import pandas; import numpy; import sklearn; print(\'Core dependencies imported successfully\')"'
            }
        }

        stage('Check ETL Script') {
            steps {
                echo 'Checking ETL pipeline script exists...'
                bat 'python -c "import src.pipeline; print(\'ETL pipeline module imported successfully\')"'
            }
        }

        stage('Check Model Scripts') {
            steps {
                echo 'Checking model script files exist...'
                bat 'if exist src\\model\\dataset.py echo dataset.py exists'
                bat 'if exist src\\model\\lstm_model.py echo lstm_model.py exists'
                bat 'if exist src\\model\\train.py echo train.py exists'
                bat 'if exist src\\model\\evaluate.py echo evaluate.py exists'
            }
        }

        stage('Archive Reports') {
            steps {
                echo 'Archiving report files if available...'
                archiveArtifacts artifacts: 'reports/*.json,reports/*.csv', allowEmptyArchive: true
            }
        }
    }

    post {
        success {
            echo 'Airman Jenkins CI pipeline completed successfully.'
        }

        failure {
            echo 'Airman Jenkins CI pipeline failed. Check the console output.'
        }

        always {
            echo 'Airman CI/CD run finished.'
        }
    }
}