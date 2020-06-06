const server=require('node-http-server');
const spawn = require("child_process").spawn;
const pythonProcess = spawn('python',["test.py"]);
 
server.deploy(
    {
        port:80
    },
   console.log( `Server on port ${server.config.port} is now up`),console.log(pythonProcess)
);

