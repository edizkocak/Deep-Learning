// globale Zustaende
static struct list *semList;

int main(int argc, char *argv[])
{
    semList = listCreate();
    if(semList == NULL)
    {
        die("listCreate");
    }

    int listenSock = Socket(AF_INET6, SOCK_STREAM, 0);
    if(listenSock == -1){
        die("listenSock");
    }

    struct sockaddr_in6 addr = {
        .sin6_family = AF_INET6,
        .sin6_port = htons(7076),
        .sin6_addr = in6addr_any,
    };

    if(bind(listenSock, (struct sock_addr*) &addr, sizeof(addr)) == -1 ){
        die("bind");
    }

    if(listen(listenSock, SOMAXCONN) == -1){
        die("listen");
    }


    while(1){
        int clientSock = accept(listenSock, NULL, NULL);
        if(clientSock == -1){
            perror("accept");
            continue;
        }

        struct client *c = clientCreate(clientSock);
        if(c == NULL){
            perror("clientCreate");
            close(clientSock); // frag nach wegen Fehlerbehandlung
            continue;
        }

        if (workerStart(clientProcess, c) == -1){
            perror("workerStart");
            close(clientSock);
            clientDestroy(c);
            continue;
        }
    }

}

int workerStart(void (*fn)(void*), void *arg)
{
    pthread_t tid;

    int err = pthread_create(&tid, NULL, fn, arg);
    if(err != 0){
        errno = err;
        return -1;
    }

    // gibt ressourcen automatisch frei nach beendigung des threads
    pthread_detach(tid);

    return 0;
}

struct client *clientCreate(int fd) 
{
    struct client *c = malloc(sizeof(*c));

    if(c == NULL){
        return NULL;
    }

    int dupfd = dup(fd);
    if(dupfd == -1){
        free(c);
        return NULL;
    }

    c->rx = fdopen(fd, "r");
    if(c->rx == NULL){
        free(c);
        close(dupfd);
        return NULL;
    }

    c->tx = fdopen(dupfd, "w");
    if(c->tx == NULL){
        fclose(c->rx);
        free(c);
        close(dupfd);
        return NULL;
    }

    c->requests = semCreate(0); // evtl was anderes
    if(c->requests == NULL){
        fclose(c->rx);
        fclose(c->tx);
        free(c);
        close(dupfd);
        return NULL;
    }

    return c;
}

void *clientProcess(void *arg){
    char buf[MAX_LINE_LEN+2];

    struct client *c = (struct client *) arg;

    aaaaaaaaaaaaaaaaaaaaaa    aaaaaaaaaaaaaaaaaaa   aaaaaaaaaaaaaaaaaaaaa\r\n\0

    cycle = false;
    while(fgets(c->rx, buf, sizeof(buf)) != EOF){
        // gesendete Zeile zu lang
        if(strlen(buf) == MAX_LINE_LEN + 1 && buf[MAX_LINE_LEN] != '\r\n'){
            cycle = true;
            continue;
        }

        // normale zeile mit normaler laenge
        if(cycle == true){
            cycle = false;
            continue;
        }

        struct request *r = malloc(sizeof(*r));

        if(r == NULL){
            reply(c->tx, buf, strerror(errno));
            free(r);
            continue;
        }

        r->client = c;

        newBuf = malloc(sizeof(char) * MAX_LINE_LEN+2);
        if(newBuf == NULL){
            reply(c->tx, buf, strerror(errno));
            free(r);
            continue;
        }

        strcpy(newBuf, buf);

        r->line = newBuf;

        pthread_t tid;
        int err = pthread_create(&tid, NULL, handleRequest, r);
        if(err != 0){
            errno = err;
            reply(c->tx, buf, strerror(errno));
            free(newBuf);
            free(r);
            continue;
        }
        pthread_detach(tid);

    }
    if(ferror(c->rx)){
        reply(c->tx, buf, strerror(errno));
    }

    P(c->requests);
    clientDestroy(c);
}


void clientDestroy(struct client* client){
    fclose(client->rx);
    fclose(client->tx);
    semDestroy(client->requests);
    free(client);
}


void handleRequest(void *arg){
    struct request *r = (struct request *) arg;

    if(line[0] == 'I'){
        if(handleI(r) != 0){
            reply(r->client->tx, r->line, strerror(errno));
        }
    }
    else if (line[0] == 'P'){
        if (handlePV(r, P) != 0){
            reply(r->client->tx, r->line, strerror(errno));
        }
    }
    else if(line[0] == 'V'){
        handlePV(r, V);
        if (handlePV(r, P) != 0){
            reply(r->client->tx, r->line, strerror(errno));
        }
    }
    else{
        reply(r->client->tx, r->line, "unknown operation");
    }

    free(r->line);
    free(r);
}

// I35
int handleI(const struct request *rq){
    int initVal;
    if(parseNumber(&(rq->line[1]), &initVal)) == -1){
        // wert ungleich 0 zurueckgeben, in der Aufgabe steht wenn ein Wert ungleich 0 zurueckgegeben wird, muss handleRequest die errno als Fehlermeldung mit reply() rausgeben
        // die errno wurde hier ja durch parseNumber gesetzt, also muessen wir es ihm nur durch einen Wert ungleich 0 signalisieren
        return -1;
    }

    SEM *sem = semCreate(initVal);
    if(sem == NULL){
        return -1;
    }

    if(listAppend(semList, sem) == -1){
        semDestroy(sem);
        return -1;
    }
    

    // im positiven Fall, gib 0 zurueck. Steht indirekt in der Aufgabe
    return 0;
}

int handlePV(const struct request *rq, void (*fn)(SEM *)){
    SEM *sem;
    if(parsePointer( &(rq->line[1]) , &sem) == -1 ){
        return -1;
    }

    if(listContains(semList, sem) == 0){
        errno = ENOENT;
        return -1;
    }

    fn(sem);

    reply(rq->client->tx, rq->line, "success");

    return 0;
}
