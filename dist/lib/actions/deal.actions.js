"use strict";
"use server";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.addUserEmailToDeal = exports.getSimilarDeals = exports.getAllDeals = exports.getDealById = exports.scrapeAndStoreDeal = void 0;
var cache_1 = require("next/cache");
var product_model_1 = __importDefault(require("../models/product.model")); // Using the Product model
var mongoose_1 = require("../mongoose");
var scraper_1 = require("../scraper");
var utils_1 = require("../utils");
var nodemailer_1 = require("../nodemailer");
function scrapeAndStoreDeal(dealUrl) {
    return __awaiter(this, void 0, void 0, function () {
        var scrapedDeal, deal, existingDeal, updatedPriceHistory, newDeal, error_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!dealUrl)
                        return [2 /*return*/];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 5, , 6]);
                    (0, mongoose_1.connectToDB)();
                    return [4 /*yield*/, (0, scraper_1.scrapeAmazonProduct)(dealUrl)];
                case 2:
                    scrapedDeal = _a.sent();
                    if (!scrapedDeal)
                        return [2 /*return*/];
                    deal = scrapedDeal;
                    return [4 /*yield*/, product_model_1.default.findOne({ url: scrapedDeal.url })];
                case 3:
                    existingDeal = _a.sent();
                    if (existingDeal) {
                        updatedPriceHistory = __spreadArray(__spreadArray([], existingDeal.priceHistory, true), [
                            { price: scrapedDeal.currentPrice, date: new Date() }
                        ], false);
                        deal = __assign(__assign({}, scrapedDeal), { priceHistory: updatedPriceHistory, lowestPrice: (0, utils_1.getLowestPrice)(updatedPriceHistory), highestPrice: (0, utils_1.getHighestPrice)(updatedPriceHistory), averagePrice: (0, utils_1.getAveragePrice)(updatedPriceHistory) });
                    }
                    return [4 /*yield*/, product_model_1.default.findOneAndUpdate({ url: scrapedDeal.url }, deal, { upsert: true, new: true })];
                case 4:
                    newDeal = _a.sent();
                    (0, cache_1.revalidatePath)("/deals/".concat(newDeal._id));
                    return [3 /*break*/, 6];
                case 5:
                    error_1 = _a.sent();
                    throw new Error("Failed to create/update deal: ".concat(error_1.message));
                case 6: return [2 /*return*/];
            }
        });
    });
}
exports.scrapeAndStoreDeal = scrapeAndStoreDeal;
function getDealById(dealId) {
    return __awaiter(this, void 0, void 0, function () {
        var deal, error_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    (0, mongoose_1.connectToDB)();
                    return [4 /*yield*/, product_model_1.default.findOne({ _id: dealId })];
                case 1:
                    deal = _a.sent();
                    if (!deal)
                        return [2 /*return*/, null];
                    return [2 /*return*/, deal];
                case 2:
                    error_2 = _a.sent();
                    console.log(error_2);
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    });
}
exports.getDealById = getDealById;
function getAllDeals() {
    return __awaiter(this, void 0, void 0, function () {
        var deals, error_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    (0, mongoose_1.connectToDB)();
                    return [4 /*yield*/, product_model_1.default.find({ discountRate: { $gt: 0 } })];
                case 1:
                    deals = _a.sent();
                    return [2 /*return*/, deals];
                case 2:
                    error_3 = _a.sent();
                    console.log(error_3);
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    });
}
exports.getAllDeals = getAllDeals;
function getSimilarDeals(dealId) {
    return __awaiter(this, void 0, void 0, function () {
        var currentDeal, similarDeals, error_4;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    (0, mongoose_1.connectToDB)();
                    return [4 /*yield*/, product_model_1.default.findById(dealId)];
                case 1:
                    currentDeal = _a.sent();
                    if (!currentDeal)
                        return [2 /*return*/, null];
                    return [4 /*yield*/, product_model_1.default.find({
                            _id: { $ne: dealId },
                            category: currentDeal.category,
                        }).limit(10)];
                case 2:
                    similarDeals = _a.sent();
                    return [2 /*return*/, similarDeals];
                case 3:
                    error_4 = _a.sent();
                    console.log(error_4);
                    return [3 /*break*/, 4];
                case 4: return [2 /*return*/];
            }
        });
    });
}
exports.getSimilarDeals = getSimilarDeals;
function addUserEmailToDeal(dealId, userEmail) {
    return __awaiter(this, void 0, void 0, function () {
        var deal, userExists, emailContent, error_5;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 6, , 7]);
                    return [4 /*yield*/, product_model_1.default.findById(dealId)];
                case 1:
                    deal = _a.sent();
                    if (!deal)
                        return [2 /*return*/];
                    userExists = deal.users.some(function (user) { return user.email === userEmail; });
                    if (!!userExists) return [3 /*break*/, 5];
                    deal.users.push({ email: userEmail });
                    return [4 /*yield*/, deal.save()];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, (0, nodemailer_1.generateEmailBody)(deal, "WELCOME")];
                case 3:
                    emailContent = _a.sent();
                    return [4 /*yield*/, (0, nodemailer_1.sendEmail)(emailContent, [userEmail])];
                case 4:
                    _a.sent();
                    _a.label = 5;
                case 5: return [3 /*break*/, 7];
                case 6:
                    error_5 = _a.sent();
                    console.log(error_5);
                    return [3 /*break*/, 7];
                case 7: return [2 /*return*/];
            }
        });
    });
}
exports.addUserEmailToDeal = addUserEmailToDeal;
