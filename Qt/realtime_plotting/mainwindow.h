#pragma once

#include <QMainWindow>
#include <QtSerialPort/QSerialPort>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void realtimeDataSlot();

private:
    Ui::MainWindow *ui;

    QSerialPort *serialPort;
    QTimer *dataTimer;

    void setUpComPort();
    void makePlot();
};
