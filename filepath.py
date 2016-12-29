import os
BasePath = os.path.dirname(os.path.dirname(__file__))

TRAINSETFILE =os.path.join(BasePath,'data/user_tag_query.10W.TRAIN')
TESTSETFILE = os.path.join(BasePath,'data/user_tag_query.10W.TEST')
TEMPFILE = os.path.join(BasePath,'temp')
RESULTFILE = os.path.join(BasePath,'data/result.csv')
TEST = os.path.join(BasePath,'temp/test')
TRAIN = os.path.join(BasePath,'temp/train')