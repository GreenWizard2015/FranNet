import wandb
import tempfile
import yaml
from functools import lru_cache

class CWBRun:
  def __init__(self, runId, api=None, tmpFolder=None):
    self._runId = runId
    self._api = api or wandb.Api()
    self._run = self._api.run(runId)
    self._tmpFolder = tmpFolder or tempfile.gettempdir()
    return
  
  @property
  @lru_cache(maxsize=1)
  def config(self):
    # load "config.yaml" from files of the run and return it as dict
    config = self._run.file('config.yaml')
    with config.download(self._tmpFolder, replace=True, exist_ok=True) as f:
      res = yaml.safe_load(f)
    # fix modifications done by wandb
    if ('model' in res) and ('value' in res['model']):
      res['model'] = res['model']['value']
    return res
  
  def models(self):
    # return list of models in the run
    res = []
    for raw in self._run.logged_artifacts():
      artifact = CWBFileArtifact(raw, self._tmpFolder, self._run)
      if artifact.name.lower().endswith('.h5'):
        res.append(artifact)
      continue

    return res
  
  @property
  def bestModel(self):
    # find 'model-latest.h5' in the run
    models = self.models()
    return next(f for f in models if f.name == 'model-latest.h5')
    
  @property
  @lru_cache(maxsize=1)
  def bestLoss(self):
    return min([x['val_loss'] for x in self.history()])
  
  @lru_cache(maxsize=1)
  def history(self):
    return self._run.scan_history()
  
  @property
  def name(self): return self._run.name

  @property
  def id(self): return self._run.id

  @property
  def fullId(self): return self._runId
# End of CWBRun

class CWBFileArtifact:
  def __init__(self, artifact, tmpFolder, run):
    self._artifact = artifact
    self._tmpFolder = tmpFolder
    self._run = run
    return
  
  def pathTo(self):
    file = self._run.use_artifact(self._artifact)
    return file.file(self._tmpFolder)
  
  @property
  def name(self):
    res = self._artifact.name
    # format: "run-{id}-{name}:{version}"
    # we need only name
    res = res.split(':')[0] # remove version
    res = res.split('-')[2:] # remove "run" and id
    return '-'.join(res)
# End of CWBFileArtifact

class CWBProject:
  def __init__(self, projectId, api=None, tmpFolder=None):
    self._projectId = projectId
    self._api = api or wandb.Api()
    self._tmpFolder = tmpFolder or tempfile.gettempdir()
    return
  
  def runs(self, filters=None):
    runs = self._api.runs(self._projectId, filters=filters)
    return [CWBRun(self._projectId + '/' + run.id, self._api, self._tmpFolder) for run in runs]
# End of CWBProject