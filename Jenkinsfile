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
                bat 'pip --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                bat 'python -m pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run ETL Pipeline') {
            steps {
                echo 'Running Airman ETL pipeline...'
                bat 'python -m src.pipeline'
            }
        }

        stage('Evaluate LSTM Model') {
            steps {
                echo 'Evaluating Airman LSTM model...'
                bat 'python -m src.model.evaluate'
            }
        }

        stage('Archive Reports') {
            steps {
                echo 'Archiving report files...'
                archiveArtifacts artifacts: 'reports/*.json,reports/*.csv', allowEmptyArchive: true
            }
        }
    }

    post {
        success {
            echo 'Airman Jenkins pipeline completed successfully.'
        }

        failure {
            echo 'Airman Jenkins pipeline failed. Check the console output.'
        }

        always {
            echo 'Airman CI/CD run finished.'
        }
    }
}