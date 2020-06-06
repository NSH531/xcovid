const express = require('express')
const app = express()
const port = 80
app.get('/data', call);
function call(req, res) {
const { execFile } = require('child_process');
const child = execFile('python3', ['J.PY'], (error, stdout, stderr) => {
  if (error) {
    throw error;
  }
  res.send(stdout);
});

 };

app.get('/', (req, res) => {
  res.send('Hello World!')
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
