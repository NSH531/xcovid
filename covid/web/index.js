var x;
x='<head><link href="https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap" rel="stylesheet"><style>.a1{font-family:"Architects Daughter", cursive;}</style></head><header><nav>  <h1 height="33" style="background-color:red;color:white">    <span style="border-style: ridge ">xcovid</span>    <span style="border-style:dotted"><a href="/data" style="color:#AADDFF">data</a></span></h1>  </nav></header>  <section>';
const dateTime = require('date-time');
 

x=x+'<P class="a1">'+dateTime(showTimeZone=true)+'</P>  <h1>xcovid19</h1>';
x=x+'<h3>Detecting corona in xrays with deep learning</h3>';
x=x+'<p>&nbsp;</p>';
x=x+'<h2><strong>Our algorithm:</strong></h2>';
x=x+'<p>we are using in CheXNet algorithm</p>';
x=x+'</section>';

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
res.send(x);

});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
