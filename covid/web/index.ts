import {PythonShell} from 'python-shell';

const express = require('express')
const app = express()
const port = 80
app.get('/data', call);
function call(req, res) {

PythonShell.run('~/covid/test.py', function (err,results) {
  if (err){
 throw err; 
} 
console.log('finished');
   res.send(results);
});
};

app.get('/', (req, res) => {
  res.send('Hello World!')
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
