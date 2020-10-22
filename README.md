# 專案說明

本專案為台灣資料科學公司以WISIDE人流偵測模組為基礎，所開發之「無線訊號推估實際人流演算模型」開放原始碼
目前階段版本進行到可以推估五個實際場域的人數。持續朝不特定場域，而是以「場域類型」的模型來前進。

# 使用方式

1. 在AI HUB 平台使用：
https://aihub.org.tw/platform/algorithm/9b39ec8c-0e93-11eb-96bc-0242ac120002/description

2. 或是您可以將專案放到自己的伺服器運行，自行修改程式碼。


# 演算法使用說明

![台灣資料科學股份有限公司](https://i.imgur.com/hC4grlU.png)

https://www.tdsc.com.tw/



##使用說明
可以透過本演算模型，以無線訊號、環境參數等資料來推測出的現場實際人數，
本演算法內部參數，會隨著本團隊的模型運作持續自我修正。


### 傳入參數說明

參數名稱  | 內容 | 型態 | 範例
------------- | -------------
wifi_person  |  WiFi裝置不重複出現次數 | int | 3500
blue_person  |  藍牙裝置不重複出現次數 | int | 6490
vendors_wifi  |  WiFi裝置廠商數量 | int | 40
vendors_blue  |  藍牙裝置廠商數量 | int | 55
signal_wifi  |  WiFi裝置訊號出現次數 | int | 6075
signal_blue  |  藍牙裝置訊號出現次數 | int | 9710
temperature  |  當日均溫 | float | 26.6
precp  |  當日雨量 | float | 5.1
event  |  是否為特殊節日(0 or 1) | int | 0
weekday  |  星期幾  1~7 (7=禮拜日) | int | 2
location  |  場域編號：1~6<br>1~6分別代表：美術館、自來水園區、天文館、木柵動物園、貓纜動物園站、貓纜貓空站 | int | 1

### 回應參數說明

參數名稱  | 內容 | 型態 | 範例
------------- | -------------
data  |  計算出的實際人數，若為負數或異常的大量，表示您輸入的參數並不合理 | int | 3500
status  |  計算結果(success,fail) | str | success


### 傳入範例
```
{
  "wifi_person": 495,
  "blue_person": 1441,
  "vendors_wifi": 45,
  "vendors_blue": 14,
  "signal_wifi": 39378,
  "signal_blue": 20939,
  "temperature": 28.9,
  "precp": 79.8,
  "event": 1,
  "weekday": 7,
  "location": 2
}

```

### 回應範例
```
{
  "data": 3339,
  "status": "success"
}

```
