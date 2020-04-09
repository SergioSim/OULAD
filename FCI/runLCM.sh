#!/bin/bash
./plcmpp/src/pLCM++ input.txt 0 .
cat *.dat > output.txt
rm *.dat
