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
                echo 'Checking model scripts exist...'
                bat 'python -c "import src.model.dataset; import src.model.lstm_model; print(\'Model modules imported successfully\')"'
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