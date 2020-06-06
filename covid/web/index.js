
let {PythonShell}= require('python-shell');
const express = require('express')
const app = express()
const port = 80
app.get('/data', call);
function call(req, res) {
let pyshell = new PythonShell('test.py', function (err, results) {
  if (err) throw err;
  // results is an array consisting of messages collected during execution
  res.send('results: %j', results);
});
 };


app.get('/', (req, res) => {
  res.send('Hello World!')
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
