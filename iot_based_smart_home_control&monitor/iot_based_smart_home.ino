#define BLYNK_PRINT Serial
 #include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>
#include <DHT.h>
 
char auth[] = "L5ag_-kce8edxG7oFpJTq6KPNDS1MiNh"; //Authentication code sent to mail
char ssid[] = "Vinzi";     // WiFi Name
char pass[] = "vinzi123";   // WiFi Password
 
#define DHTPIN 0          //Data pin of DHT11 is connected to GPIO0 ie, D3
#define DHTTYPE DHT11     
DHT dht(DHTPIN, DHTTYPE);
BlynkTimer timer;
// This function sends Arduino's up time every second to Virtual Pin (5).
void sendSensor()
{
  float h = dht.readHumidity();
  float t = dht.readTemperature();   // in degree Celsius
  if (isnan(h) || isnan(t))
 {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  Blynk.virtualWrite(V5, t);
  Blynk.virtualWrite(V6, h);
}
 
void setup()
{
  Serial.begin(9600); //set the Baud rate
 pinMode(14,OUTPUT); // Positive of LED is connected to GPIO14 ie. D5
  Blynk.begin(auth, ssid, pass);
  dht.begin();
  timer.setInterval(1000L, sendSensor);
}
 
void loop()
{
  Blynk.virtualWrite(14,HIGH);
  Blynk.virtualWrite(14,LOW);
  Blynk.virtualWrite(14,255);
  Blynk.run();
  timer.run();
}
