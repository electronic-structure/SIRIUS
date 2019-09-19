#!groovy

pipeline {
    agent {
        node {
            label 'ssl_daintvm1'
        }
    }
    environment {
        EB_CUSTOM_REPOSITORY = '/users/simonpi/jenkins/production/easybuild'
    }
    stages {
        stage('Checkout') {
            steps {
                dir('SIRIUS') {
                    checkout scm
                    echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
                }
            }
        }
        stage('Compile') {
            steps {
                dir('SIRIUS') {
                    sh '''
                           #!/bin/bash -l
                           export ENVFILE=$(realpath ci/env-gnu-gpu)
                           rm -f build-daint-gpu.out
                           rm -f build-daint-gpu.err
                           sbatch --wait ci/build-daint-gpu.sh
                           echo "---------- build-daint-gpu.out ----------"
                           cat build-daint-gpu.out
                           echo "---------- build-daint-gpu.err ----------"
                           cat build-daint-gpu.err
                           '''
                }
            }
        }
        stage('Test') {
            parallel {
                stage('Test MC') {
                    steps {
                        dir('SIRIUS') {
                            sh '''
                           cd build
                           export SIRIUS_BINARIES=$(realpath apps/dft_loop)
                           type -f ${SIRIUS_BINARIES}/sirius.scf
                           export ENVFILE=$(realpath ../ci/env-gnu-gpu)
                           rm -f sirius-mc-tests.out
                           rm -rf sirius-mc-tests.err
                           sbatch --wait ../ci/run-mc-verification.sh
                           cat sirius-mc-tests.err
                           cat sirius-mc-tests.out
                           cp *.{out,err} ../
                           '''
                        }
                    }
                }
                stage('Test GPU') {
                    steps {
                        dir('SIRIUS') {
                            sh '''
                           cd build
                           export SIRIUS_BINARIES=$(realpath apps/dft_loop)
                           type -f ${SIRIUS_BINARIES}/sirius.scf
                           export ENVFILE=$(realpath ../ci/env-gnu-gpu)
                           rm -f sirius-gpu-tests{.out,.err}
                           sbatch --wait ../ci/run-gpu-verification.sh
                           cat sirius-gpu-tests.err
                           cat sirius-gpu-tests.out
                           cp *.{out,err} ../
                           '''
                        }
                    }
                }
                stage('Test GPU Parallel') {
                    steps {
                        dir('SIRIUS') {
                            sh '''
                           cd build
                           export SIRIUS_BINARIES=$(realpath apps/dft_loop)
                           type -f ${SIRIUS_BINARIES}/sirius.scf
                           export ENVFILE=$(realpath ../ci/env-gnu-gpu)
                           rm -f sirius-gpup-tests{.out,.err}
                           sbatch --wait ../ci/run-gpup-verification.sh
                           cat sirius-gpup-tests.err
                           cat sirius-gpup-tests.out
                           cp *.{out,err} ../
                           # delete some of the heavy directories
                           cd ../
                           rm -rf build verification examples
                           '''
                        }
                    }
                }
            }
        }
    }

    post {
        always {
                archiveArtifacts artifacts: '**/*.out', fingerprint: true
                archiveArtifacts artifacts: '**/*.err', fingerprint: true
        }
    }
}
