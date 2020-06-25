const express = require('express')
const app = express()
const port = 80
app.listen(port, () => console.log(`Example app listening on port ${port}!`));

app.get('/', (req, res) => {
let ejs = require('ejs');
const fs = require('fs');

try {
    // read contents of the file
    const data = fs.readFileSync('test16june2020-1956.txt', 'UTF-8');

    // split the contents by new line
    const lines = data.split(/\r?\n/);
let html = ejs.render('<%= lines.join("<br> "); %>', {lines: lines});
  res.send(html)

   } catch (err) {
    console.error(err);
}



});

