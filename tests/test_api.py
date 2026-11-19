import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import zimage

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = patch.object(zimage, 'OUT', Path(self.tmp.name)); self.out.start()
        self.put_patch = patch.object(zimage.work, 'put'); self.put = self.put_patch.start()
        self.client = TestClient(zimage.app)
        self.headers = {'Authorization': 'Bearer ' + zimage.TOKEN}

    def tearDown(self):
        self.client.close(); self.put_patch.stop(); self.out.stop(); self.tmp.cleanup()

    def test_auth_and_validation(self):
        self.assertEqual(self.client.get('/health').status_code, 401)
        self.assertEqual(self.client.post('/jobs', headers=self.headers, json={'prompt': 'x', 'width': 513}).status_code, 422)

    def test_idempotency_and_status(self):
        payload = {'prompt':'test', 'width':512, 'height':512, 'client_request_id':'retry'}
        one = self.client.post('/jobs', headers=self.headers, json=payload)
        two = self.client.post('/jobs', headers=self.headers, json=payload)
        self.assertEqual(one.status_code, 200)
        self.assertEqual(one.json()['job_id'], two.json()['job_id'])
        self.assertEqual(self.put.call_count, 1)
        payload['steps']=9
        self.assertEqual(self.client.post('/jobs', headers=self.headers, json=payload).status_code, 409)
        self.assertEqual(self.client.get('/jobs/'+one.json()['job_id'],headers=self.headers).json()['status'], 'queued')
        self.assertEqual(self.client.get('/jobs/../../token.txt',headers=self.headers).status_code, 404)

if __name__ == '__main__': unittest.main()
