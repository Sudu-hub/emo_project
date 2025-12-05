import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/sudarshansahane1044/emo_project.mlflow")
dagshub.init(repo_owner='sudarshansahane1044', repo_name='emo_project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)