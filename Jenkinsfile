#!groovy

pipeline {
    agent {
        node {
            label 'ssl_daintvm1'
        }
    }
    environment {
        EB_CUSTOM_REPOSITORY = '/users/simonpi/jenkins/production/easybuild'
        LOGS_REPO = 'git@github.com:haampie/SIRIUS_logs.git'
        LOGS_TREE_URL = 'https://github.com/haampie/SIRIUS_logs/tree/master/'
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
                           rm -f sirius*{err,out}
                           rm -f build-daint-gpu.out
                           rm -f build-daint-gpu.err
                           sbatch --wait ci/build-daint-gpu.sh
                           echo "---------- build-daint-gpu.out ----------"
                           cat build-daint-gpu.out
                           echo "---------- build-daint-gpu.err ----------"
                           cat build-daint-gpu.err
                           # check that sirius.scf has been built
                           type -f build/apps/dft_loop/sirius.scf
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
                           sbatch --wait ../ci/run-mc-verification.sh
                           cat sirius-mc-tests.err
                           cat sirius-mc-tests.out
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
                           sbatch --wait ../ci/run-gpu-verification.sh
                           cat sirius-gpu-tests.err
                           cat sirius-gpu-tests.out
                           '''
                        }
                    }
                }
                stage('Test MC Parallel') {
                    steps {
                        dir('SIRIUS') {
                            sh '''
                           cd build
                           export SIRIUS_BINARIES=$(realpath apps/dft_loop)
                           type -f ${SIRIUS_BINARIES}/sirius.scf
                           export ENVFILE=$(realpath ../ci/env-gnu-gpu)
                           sbatch --wait ../ci/run-mcp-verification.sh
                           cat sirius-mcp-tests.err
                           cat sirius-mcp-tests.out
                           cd ../
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
                           sbatch --wait ../ci/run-gpup-verification.sh
                           cat sirius-gpup-tests.err
                           cat sirius-gpup-tests.out
                           cd ../
                           '''
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            // dir('SIRIUS') {
            //     sh '''
            //        # delete heavy directories
            //        rm -rf  build verification examples
            //        '''
            // }
            archiveArtifacts artifacts: '**/sirius*.out', fingerprint: true
            archiveArtifacts artifacts: '**/sirius*.err', fingerprint: true
            archiveArtifacts artifacts: '**/build*.out', fingerprint: true
            archiveArtifacts artifacts: '**/build*.err', fingerprint: true

            script {
                if (!changeRequest())
                    return

                dir('tmp') {
                    deleteDir()
                    withCredentials([sshUserPrivateKey(credentialsId: 'github-logs', keyFileVariable: 'SSH_KEY_PATH')]) {
                        // Clone the logs repo
                        sh """GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -i \$SSH_KEY_PATH" git clone --depth=1 ${env.LOGS_REPO} ."""

                        // Unpack the artifacts plus a readme in a folder with the name of the SHA
                        dir(pullRequest.head) {
                            unarchive mapping: ['**/**' : '.']
                            writeFile file: "readme.md", text: "Logs for ${pullRequest.url}"
                        }

                        // Push
                        sh "git add ${pullRequest.head}"
                        sh "git commit --allow-empty -m 'Add logs for ${pullRequest.url}'"
                        sh """GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -i \$SSH_KEY_PATH" git push origin master"""
                    }
                }

                pullRequest.comment("See ${env.LOGS_TREE_URL}${pullRequest.head} for the build details and benchmarks of this PR (${pullRequest.head}).")
            }
        }
    }
}
