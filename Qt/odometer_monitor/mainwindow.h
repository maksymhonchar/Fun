#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "comportutils.h"
#include "comport.h"


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
    // Update speed value.
    void spdValueUpdated();
    void timEvent();

    void on_run_btn_clicked();
    void on_stop_btn_clicked();
    void on_diameter_spnbx_valueChanged(int diameter);
    void on_MainWindow_destroyed();

    void on_resetdst_btn_clicked();

private:
    Ui::MainWindow *ui;

    const uint16_t DEFAULT_ODOM_IMP = 1000;
    const uint8_t ODOM_RATE = 200;

    uint16_t ticksCounter;

    int16_t odomImpPerSec;
    float circleLen; // mm
    double impValue; // mm per one impulse
    double totalDst_m; // m total distance
    
    // 1 sec timer to get distance.
    QTimer *oneSecTim;

    // COM port instance to receive data.
    ComPort *comPort;
    // Utils for parsing received data.
    ComPortUtils *comPortUtils;
};

#endif // MAINWINDOW_H
