
var a1=function () {  console.log('Ready')};
var x;
var f=function(port){console.log(`Example app listening on port ${port}!`)};
const fs = require('fs');
x='<head><link href="https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap" rel="stylesheet"><style>.a1{font-family:"Architects Daughter", cursive;}</style></head>';
const dateTime = require('date-time');
 

x=x+'<P class="a1">'+dateTime(showTimeZone=true)+'</P>  <h1>xcovid19</h1>';
x=x+'<h3>Detecting corona in xrays with deep learning</h3>';
x=x+'<p>&nbsp;</p>';
x=x+'<h2><strong>Our algorithm:</strong></h2>';
x=x+'<p>we are using in CheXNet algorithm</p>';
x=x+'</section>';

const express = require('express');
const http = require("http");
const app = express();
const port = 80;
app.get('/data', call);
function call(req, res) {
const fs = require('fs');

const { execFile } = require('child_process');
const child = execFile('python3', ['J.PY'], (error, stdout, stderr) => {
  if (error) {
    throw error;
  }
let data = fs.readFileSync('template.html', 'utf8');
var a=data.split('%%')[0];
var b=data.split('%%')[1];
if(b==""){
res.send(a);

}else{
res.send(a+stdout+b);

}
});
app.get('/', (req, res) => {
const fs = require('fs');

let data = fs.readFileSync('template.html', 'utf8');
var a=data.split('%%')[0];
var b=data.split('%%')[1];
if(b==""){
res.send(a);

}else{
res.send(a+x+b);

}
})

app.listen(port);
}