const server=require('node-http-server');
 
server.deploy(
    {
        port:80,
        root:'~/covid/'
    },
   console.log( `Server on port ${server.config.port} is now up`)
);
 